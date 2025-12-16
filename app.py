# app.py
from io import BytesIO
from pathlib import Path
from flask import Flask, request, render_template_string
from PIL import Image
import base64

from model import build_predictor_from_ckpt, Top1Postprocessor

app = Flask(__name__)

CKPT_PATH = Path("./out_freshguard/best_model.pth")

# Composition root for web:
PREDICTOR, _CFG = build_predictor_from_ckpt(
    CKPT_PATH,
    device=None,                   # cpu as before by default
    postprocessor=Top1Postprocessor()
)

HTML = """
<!doctype html>
<html lang="ru" data-bs-theme="dark">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FreshGuard AI</title>

  <!-- Bootstrap 5 (CDN) -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script defer src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

  <style>
    body {
      background: radial-gradient(1200px 700px at 10% 10%, rgba(13,110,253,.18), transparent 60%),
                  radial-gradient(1000px 600px at 90% 20%, rgba(25,135,84,.18), transparent 55%),
                  radial-gradient(900px 600px at 50% 90%, rgba(220,53,69,.14), transparent 60%),
                  #0b1220;
      min-height: 100vh;
    }
    .glass {
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.10);
      backdrop-filter: blur(8px);
      -webkit-backdrop-filter: blur(8px);
    }
    .brand-dot {
      width: 10px; height: 10px; border-radius: 999px;
      display: inline-block; margin-right: .5rem;
      background: linear-gradient(135deg, #0d6efd, #20c997);
      box-shadow: 0 0 0 4px rgba(13,110,253,.15);
      vertical-align: middle;
    }
    #previewWrap {
      display: none;
    }
    #previewImg {
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      background: rgba(255,255,255,.03);
      border: 1px dashed rgba(255,255,255,.18);
      border-radius: 12px;
    }
    .hint {
      color: rgba(255,255,255,.65);
      font-size: .95rem;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
  </style>
</head>

<body>
  <div class="container py-4 py-md-5">
    <div class="d-flex align-items-center justify-content-between mb-4">
      <div>
        <div class="h3 mb-1">
          <span class="brand-dot"></span>
          FreshGuard AI
        </div>
        <div class="hint">Загрузка фото → классификация → вывод результата.</div>
      </div>
      <div class="text-end d-none d-md-block">
        <div class="badge text-bg-primary">Flask</div>
        <div class="badge text-bg-success">PyTorch</div>
        <div class="badge text-bg-secondary">Image Classifier</div>
      </div>
    </div>

    <div class="row g-4">
      <!-- Left: upload -->
      <div class="col-12 col-lg-6">
        <div class="card glass shadow-sm">
          <div class="card-body p-4">
            <div class="d-flex align-items-center justify-content-between mb-3">
              <div class="h5 mb-0">Загрузить изображение</div>
              <span class="badge text-bg-info">image/*</span>
            </div>

            <form method="POST" enctype="multipart/form-data" id="uploadForm">
              <div class="mb-3">
                <label class="form-label">Файл</label>
                <input class="form-control form-control-lg" type="file" name="file" id="fileInput" accept="image/*" required>
                <div class="form-text hint mt-2">
                  Совет: лучше фото при хорошем освещении, без сильного размытия.
                </div>
              </div>

              <div class="d-grid gap-2 d-md-flex">
                <button type="submit" class="btn btn-primary btn-lg px-4" id="submitBtn">
                  Predict
                </button>
                <button type="button" class="btn btn-outline-light btn-lg" id="clearBtn">
                  Очистить
                </button>
              </div>

              <div class="mt-3" id="loadingWrap" style="display:none;">
                <div class="d-flex align-items-center justify-content-between mb-2">
                  <div class="hint">Отправка и инференс…</div>
                  <div class="spinner-border spinner-border-sm" role="status" aria-label="Loading"></div>
                </div>
                <div class="progress" style="height: 8px;">
                  <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                </div>
              </div>
            </form>
          </div>
        </div>

        <div class="card glass shadow-sm mt-4" id="previewWrap">
          <div class="card-body p-4">
            <div class="h6 mb-3">Превью</div>
            <img id="previewImg" alt="Preview">
            <div class="hint mt-2">
              <span class="mono" id="fileMeta"></span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: result -->
      <div class="col-12 col-lg-6">
        <div class="card glass shadow-sm h-100">
          <div class="card-body p-4">
            <div class="d-flex align-items-center justify-content-between mb-3">
              <div class="h5 mb-0">Результат</div>
              {% if pred %}
                <span class="badge text-bg-success">готово</span>
              {% else %}
                <span class="badge text-bg-secondary">ожидание</span>
              {% endif %}
            </div>

            {% if pred %}
              <div class="alert alert-success border-0 glass mb-3">
                <div class="d-flex align-items-start justify-content-between gap-3">
                  <div>
                    <div class="hint mb-1">Предсказание</div>
                    <div class="h4 mb-0">{{ pred }}</div>
                  </div>
                  <div class="text-end">
                    <div class="hint mb-1">Вероятность</div>
                    <div class="h4 mb-0">{{ prob }}%</div>
                  </div>
                </div>
              </div>

              <div class="mb-2 hint">Шкала уверенности (top‑1)</div>
              <div class="progress mb-4" style="height: 12px;">
                <div class="progress-bar bg-success" role="progressbar"
                     style="width: {{ prob }}%;" aria-valuenow="{{ prob }}" aria-valuemin="0" aria-valuemax="100">
                </div>
              </div>
            {% else %}
              <div class="alert alert-secondary border-0 glass mb-0">
                Загрузите изображение слева и нажмите <span class="mono">Predict</span>.
              </div>
            {% endif %}

            <hr class="my-4" style="border-color: rgba(255,255,255,.12);">

            <div class="row g-3">
              <div class="col-6">
                <div class="hint">Подсказка</div>
                <div>Проверьте резкость</div>
              </div>
              <div class="col-6">
                <div class="hint">Формат</div>
                <div>JPG/PNG/WebP</div>
              </div>
              <div class="col-12">
                <div class="hint">Приватность</div>
                <div class="note">Файл обрабатывается локально вашим сервером Flask.</div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>

    <div class="mt-4 hint">
      UI: Bootstrap компоненты/формы/карточки [web:28]. Превью: FileReader API [web:29].
    </div>
  </div>

<script>
  const fileInput = document.getElementById("fileInput");
  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const fileMeta = document.getElementById("fileMeta");
  const clearBtn = document.getElementById("clearBtn");
  const uploadForm = document.getElementById("uploadForm");
  const loadingWrap = document.getElementById("loadingWrap");
  const submitBtn = document.getElementById("submitBtn");

  function humanSize(bytes) {
    if (!bytes && bytes !== 0) return "";
    const units = ["B","KB","MB","GB"];
    let v = bytes, i = 0;
    while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
    return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
  }

  fileInput.addEventListener("change", () => {
    const f = fileInput.files && fileInput.files[0];
    if (!f) {
      previewWrap.style.display = "none";
      previewImg.removeAttribute("src");
      fileMeta.textContent = "";
      return;
    }

    // meta
    fileMeta.textContent = `${f.name} • ${humanSize(f.size)} • ${f.type || "image"}`;

    // preview via FileReader [web:29]
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewWrap.style.display = "block";
    };
    reader.readAsDataURL(f);
  });

  clearBtn.addEventListener("click", () => {
    fileInput.value = "";
    previewWrap.style.display = "none";
    previewImg.removeAttribute("src");
    fileMeta.textContent = "";
    loadingWrap.style.display = "none";
    submitBtn.disabled = false;
  });

  uploadForm.addEventListener("submit", () => {
    // чисто косметика: показать "загрузка"
    loadingWrap.style.display = "block";
    submitBtn.disabled = true;
  });
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    pred = None
    prob = None
    img_data_url = None

    if request.method == "POST":
        f = request.files["file"]

        img_bytes = f.read()
        img = Image.open(BytesIO(img_bytes)).convert("RGB")

        result = PREDICTOR.predict(img)
        pred = result.pred
        prob = result.prob_percent

        # сохранить картинку для повторного показа после POST (data URL) [web:121]
        mime = f.mimetype or "image/jpeg"
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        img_data_url = f"data:{mime};base64,{b64}"

    return render_template_string(HTML, pred=pred, prob=prob, img_data_url=img_data_url)

if __name__ == "__main__":
    app.run(debug=True)
