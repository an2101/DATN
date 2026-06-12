import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
import warnings

from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Lung Disease AI",
    page_icon="🫁",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1,h2,h3,h4 {
    color: white;
}

.small-text {
    color: #9CA3AF;
    font-size: 16px;
}


.section-title {
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# CONFIG
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = {
    "img_size": 224,
    "seg_threshold": 0.5,
    "dropout_rate": 0.5,
}

# =========================================================
# TRANSFORM
# =========================================================
def get_transforms(img_size):

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# =========================================================
# MOBILENETV2
# =========================================================
class MobileNetV2TypeClassifier(nn.Module):

    def __init__(self, num_classes=4, dropout=0.5):

        super().__init__()

        bb = mobilenet_v2(
            weights=MobileNet_V2_Weights.IMAGENET1K_V1
        )

        self.features = bb.features

        self.classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            nn.Dropout(dropout),

            nn.Linear(1280, 256),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout/2),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x

# =========================================================
# UNET
# =========================================================
class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        return self.block(x)

class UNetLungModel(nn.Module):

    def __init__(self, channels=[32,64,128,256]):

        super().__init__()

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.downsample = nn.MaxPool2d(2)

        in_channels = 3

        # Encoder
        for c in channels:

            self.encoder.append(
                ConvBlock(in_channels, c)
            )

            in_channels = c

        # Bottleneck
        self.bridge = ConvBlock(
            channels[-1],
            channels[-1] * 2
        )

        # Decoder
        for c in reversed(channels):

            self.decoder.append(
                nn.ConvTranspose2d(
                    c * 2,
                    c,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder.append(
                ConvBlock(c * 2, c)
            )

        self.out_conv = nn.Conv2d(
            channels[0],
            1,
            kernel_size=1
        )

    def forward(self, x):

        skip_feats = []

        # Encoder
        for enc in self.encoder:

            x = enc(x)

            skip_feats.append(x)

            x = self.downsample(x)

        # Bottleneck
        x = self.bridge(x)

        skip_feats = skip_feats[::-1]

        # Decoder
        for i in range(0, len(self.decoder), 2):

            x = self.decoder[i](x)

            skip = skip_feats[i // 2]

            if x.shape != skip.shape:

                x = F.interpolate(
                    x,
                    size=skip.shape[2:]
                )

            x = torch.cat([skip, x], dim=1)

            x = self.decoder[i + 1](x)

        return self.out_conv(x)

# =========================================================
# GRADCAM
# =========================================================
class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model

        self.grads = None
        self.acts = None

        self.hooks = []

        self.hooks.append(
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "acts", o)
            )
        )

        self.hooks.append(
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "grads", go[0])
            )
        )

    def __call__(self, x, class_idx=None):

        self.model.eval()

        self.model.zero_grad()

        out = self.model(x)

        if class_idx is None:

            class_idx = out.argmax(dim=1).item()

        score = out[:, class_idx]

        score.backward()

        weights = self.grads.mean(
            dim=(2,3),
            keepdim=True
        )

        cam = (weights * self.acts).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam

    def remove_hooks(self):

        for h in self.hooks:
            h.remove()

# =========================================================
# HEATMAP
# =========================================================
def apply_colormap(img, cam, alpha=0.4):

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    ) / 255.0

    overlay = alpha * heatmap + (1 - alpha) * img

    return np.uint8(255 * overlay)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_models():

    unet_path = "Model/unet_model.pth"

    mobile_path = "Model/mobilenetv2_best.pth"

    lung_unet = UNetLungModel().to(DEVICE)

    mobilenet = MobileNetV2TypeClassifier(
        num_classes=4,
        dropout=CFG["dropout_rate"]
    ).to(DEVICE)

    lung_unet.load_state_dict(
        torch.load(unet_path, map_location=DEVICE)
    )

    mobilenet.load_state_dict(
        torch.load(mobile_path, map_location=DEVICE)
    )

    lung_unet.eval()
    mobilenet.eval()

    return lung_unet, mobilenet

# =========================================================
# LOAD
# =========================================================
lung_unet, mobilenet = load_models()

# =========================================================
# UI
# =========================================================
st.title("🫁 Lung Disease Classification AI")

st.markdown(
    """
    <p class='small-text'>
    UNet Lung Segmentation + MobileNetV2 + Grad-CAM Visualization
    </p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg","jpeg","png"]
)

# =========================================================
# PROCESS
# =========================================================
if uploaded_file:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    if img is None:

        st.error("Cannot read image")

        st.stop()

    img_rgb = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    # =====================================================
    # UNET
    # =====================================================
    img_resize = cv2.resize(
        img_rgb,
        (CFG["img_size"], CFG["img_size"])
    )

    tfm = get_transforms(CFG["img_size"])

    img_tensor = tfm(
        image=img_resize
    )["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        mask = torch.sigmoid(
            lung_unet(img_tensor)
        )

        mask = (
            mask > CFG["seg_threshold"]
        ).float()

    mask_np = mask.squeeze().cpu().numpy()

    # =====================================================
    # CROP
    # =====================================================
    rows = np.any(mask_np, axis=1)

    cols = np.any(mask_np, axis=0)

    if not rows.any():

        crop_resize = cv2.resize(
            img_rgb,
            (224,224)
        )

    else:

        r0, r1 = np.where(rows)[0][[0,-1]]

        c0, c1 = np.where(cols)[0][[0,-1]]

        pad = 0.1

        H, W = mask_np.shape

        pr = int((r1-r0) * pad)
        pc = int((c1-c0) * pad)

        r0 = max(0, r0-pr)
        r1 = min(H, r1+pr)

        c0 = max(0, c0-pc)
        c1 = min(W, c1+pc)

        crop = img_resize[r0:r1, c0:c1]

        crop_resize = cv2.resize(
            crop,
            (224,224)
        )

    # =====================================================
    # CLASSIFICATION
    # =====================================================
    clf_tfm = get_transforms(224)

    x = clf_tfm(
        image=crop_resize
    )["image"].unsqueeze(0).to(DEVICE)

    class_names = [
        "NORMAL",
        "BACTERIAL",
        "VIRAL",
        "COVID"
    ]

    output = mobilenet(x)

    probs = torch.softmax(output, dim=1)

    prob, pred = torch.max(probs, dim=1)

    label = class_names[pred.item()]

    prob = prob.item()

    # =====================================================
    # GRADCAM
    # =====================================================
    target_layer = mobilenet.features[-1]

    gradcam = GradCAM(
        mobilenet,
        target_layer
    )

    x_cam = x.clone().requires_grad_(True)

    cam = gradcam(
        x_cam,
        class_idx=pred.item()
    )

    img_cam = crop_resize.astype(np.float32) / 255.0

    cam_overlay = apply_colormap(
        img_cam,
        cam
    )

    gradcam.remove_hooks()

    # =====================================================
    # DISPLAY IMAGE
    # =====================================================
    st.markdown(
        "<div class='section-title'>Visualization</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            img_rgb,
            caption="Original Image",
            use_container_width=True
        )

        st.image(
            crop_resize,
            caption="Cropped Lung",
            use_container_width=True
        )

    with col2:

        st.image(
            mask_np,
            caption="UNet Mask",
            use_container_width=True
        )

        st.image(
            cam_overlay,
            caption="Grad-CAM",
            use_container_width=True
        )

    # =====================================================
    # RESULT
    # =====================================================
    st.markdown("## Prediction Result")

    result_color = "#22C55E"

    if label != "NORMAL":
        result_color = "#EF4444"

    st.markdown(
        f"""
     <div style="
        background: linear-gradient(135deg,#111827,#1F2937);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #374151;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
     ">

        <h1 style="
            color:{result_color};
            margin-bottom:10px;
            font-size:48px;
        ">
            {label}
        </h1>

        <h3 style="
            color:white;
            font-weight:normal;
        ">
            Confidence: {prob*100:.2f}%
        </h3>

     </div>
     """,
     unsafe_allow_html=True
     )

    # =====================================================
    # PROBABILITIES
    # =====================================================
    st.markdown(
        "<div class='section-title'>Class Probabilities</div>",
        unsafe_allow_html=True
    )

    probs_np = probs.squeeze().detach().cpu().numpy()

    for i, cls in enumerate(class_names):

        p = float(probs_np[i])

        st.write(f"{cls}: {p*100:.2f}%")

        st.progress(p)