(() => {
  const STORAGE_KEY = "mls.uiLanguage";
  const DEFAULT_LOCALE = "zh-CN";
  const SUPPORTED_LOCALES = ["zh-CN", "ja-JP", "en-US"];

  const messages = {
    "zh-CN": {
      "app.title": "漫画本地化工作台",
      "app.subtitle": "导入图片，自动识别并生成连贯的本地化版本",
      "app.github": "打开 GitHub",
      "language.label": "界面语言",
      "nav.main": "主导航",
      "nav.workspace": "新建项目",
      "nav.history": "任务记录",
      "nav.models": "模型管理",
      "nav.settings": "设置",
      "privacy.local.title": "本地推理模式",
      "privacy.local.detail": "图片和文字不离开设备",
      "privacy.local.notice": "所有处理均在本机完成，不上传图片",
      "privacy.online.title": "在线翻译模式",
      "privacy.online.detail": "仅发送识别后的文字",
      "privacy.online.notice": "图片与排版留在本机；仅 OCR 文字会发送到所选翻译 API",
      "setup.title": "项目设置",
      "setup.quickStart": "查看快速上手",
      "setup.source": "源图片文件夹",
      "setup.source.placeholder": "选择包含漫画图片的文件夹",
      "setup.output": "输出目录",
      "setup.output.placeholder": "不会覆盖源图片",
      "setup.pickFolder": "选择文件夹",
      "setup.target": "目标语言",
      "target.zhHans": "简体中文",
      "target.zhHant": "繁体中文",
      "target.english": "英语",
      "setup.reviewed": "高级：导入已审校 transcript.json",
      "setup.reviewed.placeholder": "可选；导入后跳过 OCR 和翻译，直接安全重排",
      "pipeline.title": "本地化流程",
      "pipeline.waiting": "等待开始",
      "pipeline.complete": "已完成",
      "pipeline.detect": "文字检测",
      "pipeline.detect.detail": "定位文字区域",
      "pipeline.ocr": "漫画 OCR",
      "pipeline.ocr.detail": "读取日文内容",
      "pipeline.translate": "连贯翻译",
      "pipeline.translate.detail": "参考前后页面",
      "pipeline.render": "原位替换",
      "pipeline.render.detail": "保留原始尺寸",
      "preview.title": "页面预览",
      "preview.none": "尚未载入页面",
      "preview.mode": "预览模式",
      "preview.source": "原图",
      "preview.output": "译文",
      "preview.compare": "对比",
      "preview.empty.title": "选择源文件夹后开始任务",
      "preview.empty.detail": "完成的页面会在这里提供原图与译文对比",
      "preview.source.alt": "源漫画页面预览",
      "preview.output.alt": "本地化页面预览",
      "preview.output.waiting": "等待生成译文页面",
      "preview.output.unavailable": "译文页面暂不可用",
      "preview.output.waitOcr": "等待 OCR 完成",
      "preview.output.waitTranslation": "等待连贯翻译完成",
      "preview.output.failed": "任务失败，暂无译文页面",
      "preview.previous": "上一页",
      "preview.next": "下一页",
      "preview.page": "第 {page} 页",
      "models.status": "模型状态",
      "models.refresh": "刷新模型状态",
      "models.prepareMissing": "准备缺失模型",
      "models.preparing": "准备模型…",
      "models.ready": "已就绪",
      "models.missing": "未下载",
      "models.optional": "可选",
      "models.optionalCurrent": "当前可选",
      "models.notRequired": "当前后端不需要",
      "models.unavailable": "模型状态不可用",
      "models.prepareCount": "准备 {count} 个缺失模型",
      "models.allReady": "所需模型已就绪",
      "models.prepared": "模型已准备完成",
      "models.role.paddleocr": "文字检测与行级识别",
      "models.role.manga-ocr": "日文漫画识别",
      "models.role.lama": "文字区域背景补全",
      "models.role.hy-mt2": "连贯翻译",
      "translation.title": "翻译设置",
      "translation.quality": "处理质量",
      "translation.quality.quality": "质量优先（9B 分阶段审校＋Hy-MT2＋LaMa）",
      "translation.quality.fast": "轻量快速（单模型翻译＋传统补画）",
      "translation.format": "输出格式",
      "translation.format.webp": "无损 WebP（推荐，体积较小）",
      "translation.format.png": "无损 PNG（兼容优先）",
      "translation.ocr": "OCR 后端",
      "translation.ocr.hybrid": "混合质量 OCR（推荐）",
      "translation.ocr.builtin": "内置专用 OCR（快速）",
      "translation.ocr.ollama": "仅 Ollama 视觉模型（实验）",
      "translation.inference": "推理后端",
      "translation.inference.ollama": "Ollama 本地推理（推荐）",
      "translation.inference.builtin": "内置 Hy-MT2（快速）",
      "translation.inference.online": "在线兼容 API",
      "translation.story": "故事连贯模式",
      "translation.story.detail": "参考相邻页面调整人名和语气",
      "translation.context": "上下文页数",
      "translation.sfx": "保留拟声词",
      "translation.sfx.detail": "仅替换对白和旁白",
      "translation.start": "开始本地化",
      "translation.start.missingModels": "请先准备缺失模型",
      "translation.start.missingName": "请填写推理模型名称",
      "job.notStarted": "尚未开始任务",
      "job.selectFolders": "选择源文件夹和输出目录后即可开始",
      "job.local": "本地任务",
      "job.queued": "等待开始",
      "job.preparing": "准备本地化",
      "job.ocr": "正在识别第 {current}/{total} 页",
      "job.translate": "正在翻译第 {current}/{total} 页",
      "job.render": "正在生成第 {current}/{total} 页",
      "job.complete": "本地化完成",
      "job.failed": "任务失败",
      "job.pause": "暂停",
      "job.pauseUnsupported": "当前版本不支持安全暂停",
      "history.title": "任务记录",
      "history.detail": "任务状态会保存在本机应用目录。",
      "history.refresh": "刷新",
      "history.empty": "还没有任务记录",
      "status.queued": "等待中",
      "status.running": "进行中",
      "status.complete": "已完成",
      "status.failed": "失败",
      "modelPage.title": "模型管理",
      "modelPage.detail": "默认优先从 ModelScope 下载；无等价模型时明确回退。",
      "modelPage.prepareAll": "准备全部模型",
      "settings.title": "设置",
      "settings.detail": "后端配置保存在本机；API 密钥仅保留在当前服务进程或环境变量中。",
      "settings.check": "测试推理连接",
      "settings.ocr.hybrid": "Paddle 精确框＋Ollama 整页语义（推荐）",
      "settings.ocr.builtin": "PaddleOCR＋Manga OCR（快速）",
      "settings.ocr.ollama": "仅 Ollama 视觉 OCR（实验）",
      "settings.inference.ollama": "Ollama 本地推理（最高 9B）",
      "settings.inference.builtin": "内置 Hy-MT2（快速模式）",
      "settings.inference.online": "在线 OpenAI 兼容 API",
      "settings.builtin.title": "内置轻量翻译模型",
      "settings.builtin.detail": "使用 ModelScope 下载的 Hy-MT2 1.8B，适合低配置快速初译；质量模式推荐本机 Ollama。",
      "settings.ollama.url": "Ollama 地址",
      "settings.ollama.model": "翻译模型名称",
      "settings.ollama.ocrModel": "视觉 OCR 模型名称",
      "settings.ollama.detail": "质量模式只使用所选的 9B 或更小 Ollama 模型：先生成草稿，再与 Hy-MT2 独立候选做分阶段上下文审校，不下载更大的模型。",
      "settings.online.url": "兼容 API 地址",
      "settings.online.model": "模型名称",
      "settings.online.model.placeholder": "输入服务商提供的模型 ID",
      "settings.online.key": "API 密钥",
      "settings.online.key.placeholder": "可使用 MLS_ONLINE_API_KEY",
      "settings.online.key.configured": "当前会话已配置；留空保持不变",
      "settings.online.detail": "在线模式只发送 OCR 文本、前文和术语表，不上传漫画图片。",
      "settings.inference.notChecked": "尚未测试连接",
      "settings.inference.checking": "正在测试连接…",
      "settings.inference.connected": "已连接，发现 {count} 个模型",
      "settings.device": "计算设备",
      "settings.device.auto": "自动选择",
      "settings.modelscope": "ModelScope 优先",
      "settings.modelscope.detail": "适合中国大陆网络环境",
      "settings.home": "应用数据目录：{path}",
      "folder.manual": "{message}；也可以直接输入绝对路径。",
      "quickStart.close": "关闭",
      "quickStart.title": "三步快速上手",
      "quickStart.step1": "先在“模型管理”准备所需权重。",
      "quickStart.step2": "选择源图片文件夹和独立输出目录。",
      "quickStart.step3": "点击“开始本地化”，任务可在记录页继续查看。",
      "quickStart.detail": "首次下载约需 5 GB；图片与结果不会上传。"
    },
    "ja-JP": {
      "app.title": "マンガローカライズ作業台", "app.subtitle": "画像を読み込み、文字を認識して一貫したローカライズ版を生成します", "app.github": "GitHub を開く", "language.label": "表示言語",
      "nav.main": "メインナビゲーション", "nav.workspace": "新規プロジェクト", "nav.history": "ジョブ履歴", "nav.models": "モデル管理", "nav.settings": "設定",
      "privacy.local.title": "ローカル推論", "privacy.local.detail": "画像とテキストは端末外へ送信されません", "privacy.local.notice": "すべて端末上で処理され、画像はアップロードされません", "privacy.online.title": "オンライン翻訳", "privacy.online.detail": "認識済みテキストのみ送信", "privacy.online.notice": "画像と組版は端末内に保持し、OCR テキストのみ翻訳 API に送信します",
      "setup.title": "プロジェクト設定", "setup.quickStart": "クイックスタート", "setup.source": "元画像フォルダー", "setup.source.placeholder": "マンガ画像を含むフォルダーを選択", "setup.output": "出力先", "setup.output.placeholder": "元画像は上書きされません", "setup.pickFolder": "フォルダー選択", "setup.target": "翻訳先言語", "target.zhHans": "簡体字中国語", "target.zhHant": "繁体字中国語", "target.english": "英語", "setup.reviewed": "詳細：校正済み transcript.json を読み込む", "setup.reviewed.placeholder": "任意。OCR と翻訳を省略し、安全に再配置します",
      "pipeline.title": "ローカライズ工程", "pipeline.waiting": "開始待ち", "pipeline.complete": "完了", "pipeline.detect": "文字検出", "pipeline.detect.detail": "文字領域を特定", "pipeline.ocr": "マンガ OCR", "pipeline.ocr.detail": "日本語を読み取り", "pipeline.translate": "文脈翻訳", "pipeline.translate.detail": "前後ページを参照", "pipeline.render": "位置を維持して置換", "pipeline.render.detail": "元の寸法を維持",
      "preview.title": "ページプレビュー", "preview.none": "ページ未読込", "preview.mode": "プレビューモード", "preview.source": "原稿", "preview.output": "翻訳", "preview.compare": "比較", "preview.empty.title": "元画像フォルダーを選択して開始", "preview.empty.detail": "完了したページの原稿と翻訳をここで比較できます", "preview.source.alt": "元マンガページのプレビュー", "preview.output.alt": "ローカライズ済みページのプレビュー", "preview.output.waiting": "翻訳ページの生成待ち", "preview.output.unavailable": "翻訳ページを表示できません", "preview.output.waitOcr": "OCR の完了待ち", "preview.output.waitTranslation": "翻訳の完了待ち", "preview.output.failed": "ジョブ失敗のため翻訳ページはありません", "preview.previous": "前のページ", "preview.next": "次のページ", "preview.page": "{page} ページ目",
      "models.status": "モデル状態", "models.refresh": "モデル状態を更新", "models.prepareMissing": "不足モデルを準備", "models.preparing": "モデルを準備中…", "models.ready": "準備済み", "models.missing": "未ダウンロード", "models.optional": "任意", "models.optionalCurrent": "現在は任意", "models.notRequired": "現在のバックエンドでは不要", "models.unavailable": "モデル状態を取得できません", "models.prepareCount": "不足モデル {count} 件を準備", "models.allReady": "必要なモデルは準備済みです", "models.prepared": "モデルの準備が完了しました", "models.role.paddleocr": "文字検出と行単位認識", "models.role.manga-ocr": "日本語マンガ認識", "models.role.lama": "文字領域の背景補完", "models.role.hy-mt2": "文脈翻訳",
      "translation.title": "翻訳設定", "translation.quality": "処理品質", "translation.quality.quality": "品質優先（9B 段階校正＋Hy-MT2＋LaMa）", "translation.quality.fast": "軽量・高速（単一モデル翻訳＋従来型補完）", "translation.format": "出力形式", "translation.format.webp": "ロスレス WebP（推奨・小容量）", "translation.format.png": "ロスレス PNG（互換性優先）", "translation.ocr": "OCR バックエンド", "translation.ocr.hybrid": "ハイブリッド高品質 OCR（推奨）", "translation.ocr.builtin": "内蔵専用 OCR（高速）", "translation.ocr.ollama": "Ollama 視覚モデルのみ（実験的）", "translation.inference": "推論バックエンド", "translation.inference.ollama": "Ollama ローカル推論（推奨）", "translation.inference.builtin": "内蔵 Hy-MT2（高速）", "translation.inference.online": "オンライン互換 API", "translation.story": "ストーリー文脈モード", "translation.story.detail": "隣接ページを参照して人名と口調を調整", "translation.context": "参照ページ数", "translation.sfx": "効果音を保持", "translation.sfx.detail": "セリフとナレーションのみ置換", "translation.start": "ローカライズ開始", "translation.start.missingModels": "先に不足モデルを準備してください", "translation.start.missingName": "推論モデル名を入力してください",
      "job.notStarted": "ジョブはまだ開始されていません", "job.selectFolders": "元画像フォルダーと出力先を選択してください", "job.local": "ローカルジョブ", "job.queued": "開始待ち", "job.preparing": "ローカライズを準備中", "job.ocr": "{current}/{total} ページを認識中", "job.translate": "{current}/{total} ページを翻訳中", "job.render": "{current}/{total} ページを生成中", "job.complete": "ローカライズ完了", "job.failed": "ジョブに失敗しました", "job.pause": "一時停止", "job.pauseUnsupported": "現在のバージョンでは安全な一時停止に対応していません",
      "history.title": "ジョブ履歴", "history.detail": "ジョブ状態はローカルのアプリデータに保存されます。", "history.refresh": "更新", "history.empty": "ジョブ履歴はありません", "status.queued": "待機中", "status.running": "実行中", "status.complete": "完了", "status.failed": "失敗",
      "modelPage.title": "モデル管理", "modelPage.detail": "ModelScope を優先し、同等モデルがない場合のみ明示的にフォールバックします。", "modelPage.prepareAll": "全モデルを準備",
      "settings.title": "設定", "settings.detail": "バックエンド設定は端末内に保存され、API キーは現在のプロセスまたは環境変数にのみ保持されます。", "settings.check": "推論接続をテスト", "settings.ocr.hybrid": "Paddle の精密枠＋Ollama ページ全体の意味解析（推奨）", "settings.ocr.builtin": "PaddleOCR＋Manga OCR（高速）", "settings.ocr.ollama": "Ollama 視覚 OCR のみ（実験的）", "settings.inference.ollama": "Ollama ローカル推論（最大 9B）", "settings.inference.builtin": "内蔵 Hy-MT2（高速モード）", "settings.inference.online": "オンライン OpenAI 互換 API", "settings.builtin.title": "内蔵軽量翻訳モデル", "settings.builtin.detail": "ModelScope から取得する Hy-MT2 1.8B を使用します。低スペック環境の初訳向けで、品質モードではローカル Ollama を推奨します。", "settings.ollama.url": "Ollama URL", "settings.ollama.model": "翻訳モデル名", "settings.ollama.ocrModel": "視覚 OCR モデル名", "settings.ollama.detail": "品質モードでは選択した 9B 以下の Ollama モデルのみを使用し、下訳後に Hy-MT2 の独立候補と段階的な文脈校正を行います。より大きなモデルは取得しません。", "settings.online.url": "互換 API URL", "settings.online.model": "モデル名", "settings.online.model.placeholder": "プロバイダーのモデル ID を入力", "settings.online.key": "API キー", "settings.online.key.placeholder": "MLS_ONLINE_API_KEY も使用できます", "settings.online.key.configured": "現在のセッションで設定済み。空欄なら維持します", "settings.online.detail": "オンラインモードでは OCR テキスト、前文、用語集のみ送信し、マンガ画像は送信しません。", "settings.inference.notChecked": "接続は未テストです", "settings.inference.checking": "接続をテスト中…", "settings.inference.connected": "接続済み：{count} モデルを検出", "settings.device": "計算デバイス", "settings.device.auto": "自動選択", "settings.modelscope": "ModelScope を優先", "settings.modelscope.detail": "中国本土のネットワーク環境向け", "settings.home": "アプリデータ：{path}", "folder.manual": "{message}。絶対パスを直接入力することもできます。",
      "quickStart.close": "閉じる", "quickStart.title": "3 ステップで開始", "quickStart.step1": "「モデル管理」で必要な重みを準備します。", "quickStart.step2": "元画像フォルダーと別の出力先を選択します。", "quickStart.step3": "「ローカライズ開始」を押します。進行状況は履歴ページで確認できます。", "quickStart.detail": "初回ダウンロードは約 5 GB です。画像と結果はアップロードされません。"
    },
    "en-US": {
      "app.title": "Manga Localization Workspace", "app.subtitle": "Import images, recognize text, and produce a coherent localized edition", "app.github": "Open GitHub", "language.label": "Interface language",
      "nav.main": "Main navigation", "nav.workspace": "New project", "nav.history": "Job history", "nav.models": "Model manager", "nav.settings": "Settings",
      "privacy.local.title": "Local inference", "privacy.local.detail": "Images and text stay on this device", "privacy.local.notice": "All processing runs locally. Images are never uploaded.", "privacy.online.title": "Online translation", "privacy.online.detail": "Only recognized text is sent", "privacy.online.notice": "Images and layout stay local; only OCR text is sent to the selected translation API.",
      "setup.title": "Project setup", "setup.quickStart": "Quick start", "setup.source": "Source image folder", "setup.source.placeholder": "Select a folder containing manga images", "setup.output": "Output folder", "setup.output.placeholder": "Source images will not be overwritten", "setup.pickFolder": "Choose folder", "setup.target": "Target language", "target.zhHans": "Simplified Chinese", "target.zhHant": "Traditional Chinese", "target.english": "English", "setup.reviewed": "Advanced: import a reviewed transcript.json", "setup.reviewed.placeholder": "Optional; skips OCR and translation, then safely reflows the reviewed text",
      "pipeline.title": "Localization pipeline", "pipeline.waiting": "Waiting to start", "pipeline.complete": "Complete", "pipeline.detect": "Text detection", "pipeline.detect.detail": "Locate text regions", "pipeline.ocr": "Manga OCR", "pipeline.ocr.detail": "Read Japanese text", "pipeline.translate": "Coherent translation", "pipeline.translate.detail": "Reference nearby pages", "pipeline.render": "In-place replacement", "pipeline.render.detail": "Preserve original dimensions",
      "preview.title": "Page preview", "preview.none": "No page loaded", "preview.mode": "Preview mode", "preview.source": "Source", "preview.output": "Translation", "preview.compare": "Compare", "preview.empty.title": "Select a source folder to begin", "preview.empty.detail": "Completed pages will show source and translation side by side here", "preview.source.alt": "Source manga page preview", "preview.output.alt": "Localized page preview", "preview.output.waiting": "Waiting for the translated page", "preview.output.unavailable": "Translated page is temporarily unavailable", "preview.output.waitOcr": "Waiting for OCR to finish", "preview.output.waitTranslation": "Waiting for translation to finish", "preview.output.failed": "The job failed; no translated page is available", "preview.previous": "Previous page", "preview.next": "Next page", "preview.page": "Page {page}",
      "models.status": "Model status", "models.refresh": "Refresh model status", "models.prepareMissing": "Prepare missing models", "models.preparing": "Preparing models…", "models.ready": "Ready", "models.missing": "Not downloaded", "models.optional": "Optional", "models.optionalCurrent": "Currently optional", "models.notRequired": "Not required by the current backend", "models.unavailable": "Model status unavailable", "models.prepareCount": "Prepare {count} missing models", "models.allReady": "Required models are ready", "models.prepared": "Models are ready", "models.role.paddleocr": "Text detection and line recognition", "models.role.manga-ocr": "Japanese manga recognition", "models.role.lama": "Text-region background reconstruction", "models.role.hy-mt2": "Coherent translation",
      "translation.title": "Translation settings", "translation.quality": "Processing quality", "translation.quality.quality": "Quality first (staged 9B review + Hy-MT2 + LaMa)", "translation.quality.fast": "Lightweight and fast (single-model translation + classic inpainting)", "translation.format": "Output format", "translation.format.webp": "Lossless WebP (recommended, smaller)", "translation.format.png": "Lossless PNG (best compatibility)", "translation.ocr": "OCR backend", "translation.ocr.hybrid": "Hybrid quality OCR (recommended)", "translation.ocr.builtin": "Built-in specialized OCR (fast)", "translation.ocr.ollama": "Ollama vision only (experimental)", "translation.inference": "Inference backend", "translation.inference.ollama": "Local Ollama inference (recommended)", "translation.inference.builtin": "Built-in Hy-MT2 (fast)", "translation.inference.online": "Online compatible API", "translation.story": "Story context mode", "translation.story.detail": "Use nearby pages to align names and tone", "translation.context": "Context pages", "translation.sfx": "Preserve sound effects", "translation.sfx.detail": "Replace dialogue and narration only", "translation.start": "Start localization", "translation.start.missingModels": "Prepare the missing models first", "translation.start.missingName": "Enter an inference model name",
      "job.notStarted": "No job has started", "job.selectFolders": "Select source and output folders to begin", "job.local": "Local job", "job.queued": "Waiting to start", "job.preparing": "Preparing localization", "job.ocr": "Recognizing page {current}/{total}", "job.translate": "Translating page {current}/{total}", "job.render": "Generating page {current}/{total}", "job.complete": "Localization complete", "job.failed": "Job failed", "job.pause": "Pause", "job.pauseUnsupported": "Safe pause is not available in this version",
      "history.title": "Job history", "history.detail": "Job status is stored in the local app data folder.", "history.refresh": "Refresh", "history.empty": "No jobs yet", "status.queued": "Queued", "status.running": "Running", "status.complete": "Complete", "status.failed": "Failed",
      "modelPage.title": "Model manager", "modelPage.detail": "Downloads prefer ModelScope and explicitly fall back only when no equivalent model exists.", "modelPage.prepareAll": "Prepare all models",
      "settings.title": "Settings", "settings.detail": "Backend settings stay local. API keys live only in the current process or environment variables.", "settings.check": "Test inference connection", "settings.ocr.hybrid": "Precise Paddle boxes + Ollama whole-page semantics (recommended)", "settings.ocr.builtin": "PaddleOCR + Manga OCR (fast)", "settings.ocr.ollama": "Ollama vision OCR only (experimental)", "settings.inference.ollama": "Local Ollama inference (up to 9B)", "settings.inference.builtin": "Built-in Hy-MT2 (fast mode)", "settings.inference.online": "Online OpenAI-compatible API", "settings.builtin.title": "Built-in lightweight translation model", "settings.builtin.detail": "Uses the ModelScope-hosted Hy-MT2 1.8B for fast drafts on modest hardware. Local Ollama is recommended for quality mode.", "settings.ollama.url": "Ollama URL", "settings.ollama.model": "Translation model name", "settings.ollama.ocrModel": "Vision OCR model name", "settings.ollama.detail": "Quality mode uses only the selected 9B-or-smaller Ollama model: it drafts first, then performs staged contextual review against an independent Hy-MT2 candidate. No larger model is downloaded.", "settings.online.url": "Compatible API URL", "settings.online.model": "Model name", "settings.online.model.placeholder": "Enter the model ID supplied by your provider", "settings.online.key": "API key", "settings.online.key.placeholder": "You can also use MLS_ONLINE_API_KEY", "settings.online.key.configured": "Configured for this session; leave blank to keep it", "settings.online.detail": "Online mode sends only OCR text, prior context, and the glossary. Manga images are never uploaded.", "settings.inference.notChecked": "Connection not tested", "settings.inference.checking": "Testing connection…", "settings.inference.connected": "Connected; found {count} models", "settings.device": "Compute device", "settings.device.auto": "Auto-select", "settings.modelscope": "Prefer ModelScope", "settings.modelscope.detail": "Optimized for network access in mainland China", "settings.home": "App data: {path}", "folder.manual": "{message}; you can also enter an absolute path directly.",
      "quickStart.close": "Close", "quickStart.title": "Start in three steps", "quickStart.step1": "Prepare the required weights in Model manager.", "quickStart.step2": "Choose the source image folder and a separate output folder.", "quickStart.step3": "Select Start localization; follow progress later in Job history.", "quickStart.detail": "The first download is about 5 GB. Images and results are never uploaded."
    }
  };

  function normalizeLocale(locale) {
    const value = String(locale || "").toLowerCase();
    if (value.startsWith("ja")) return "ja-JP";
    if (value.startsWith("en")) return "en-US";
    if (value.startsWith("zh")) return "zh-CN";
    return DEFAULT_LOCALE;
  }

  function initialLocale() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (SUPPORTED_LOCALES.includes(stored)) return stored;
    } catch {}
    return normalizeLocale(navigator.languages?.[0] || navigator.language);
  }

  let locale = initialLocale();

  function t(key, parameters = {}) {
    const template = messages[locale]?.[key] ?? messages[DEFAULT_LOCALE]?.[key] ?? key;
    return template.replace(/\{(\w+)\}/g, (_match, name) => parameters[name] ?? `{${name}}`);
  }

  function apply(root = document) {
    root.querySelectorAll("[data-i18n]").forEach((element) => {
      element.textContent = t(element.dataset.i18n);
    });
    for (const attribute of ["placeholder", "aria-label", "title", "alt"]) {
      const datasetName = `i18n${attribute.split("-").map((part) => part[0].toUpperCase() + part.slice(1)).join("")}`;
      root.querySelectorAll(`[data-i18n-${attribute}]`).forEach((element) => {
        element.setAttribute(attribute, t(element.dataset[datasetName]));
      });
    }
    document.documentElement.lang = locale;
    document.title = t("app.title");
  }

  function setLocale(nextLocale) {
    locale = normalizeLocale(nextLocale);
    try { localStorage.setItem(STORAGE_KEY, locale); } catch {}
    apply();
    window.dispatchEvent(new CustomEvent("mls:localechange", { detail: { locale } }));
  }

  window.MLS_I18N = { apply, getLocale: () => locale, setLocale, supportedLocales: SUPPORTED_LOCALES, t };
})();
