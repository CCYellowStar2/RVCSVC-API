# MSST 模型按需下载 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在用户选择 MSST BS-Roformer 时，自动下载缺失的权重后继续完成分离。

**Architecture:** 由 `msst_backend.py` 集中管理模型 URL、完整性判断和下载，因此 RVC 与 SVC 都通过既有的 `separate_msst_audio()` 获得相同行为。下载写入同目录临时文件，完成后用 `replace()` 发布为目标权重，防止中途失败时留下不完整文件。

**Tech Stack:** Python 3.10、标准库 `urllib.request`、`unittest`。

## Global Constraints

- MSST 权重仅在选择 `MSST BS-Roformer` 且本地缺失时下载。
- VR-HP5 的启动期 ModelScope 下载行为不得改变。
- YAML 配置文件继续随仓库分发，不下载。
- 不将模型、音频或测试目录提交到 GitHub。

---

### Task 1: 为缺失 MSST 权重实现安全的按需下载

**Files:**
- Modify: `D:/对比/RVCSVC-API/msst_backend.py`
- Modify: `D:/对比/RVCSVC-API-github/msst_backend.py`
- Test: `D:/对比/RVCSVC-API/tests/test_msst_backend.py`

**Interfaces:**
- Consumes: `asset_paths(root, model_selection) -> tuple[Path, Path]`
- Produces: `ensure_model_downloaded(model_path: Path, opener: Callable = urlopen) -> Path`
- Uses: `validate_assets()` before `MSSeparator` is initialized.

- [ ] **Step 1: Write the failing test**

```python
@patch("msst_backend.urlopen")
def test_validate_assets_downloads_missing_model(self, urlopen):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _, config_path = msst_backend.asset_paths(root)
        config_path.parent.mkdir(parents=True)
        config_path.touch()
        urlopen.return_value.__enter__.return_value.read.side_effect = [b"weight", b""]

        model_path, _ = msst_backend.validate_assets(root)

        self.assertEqual(model_path.read_bytes(), b"weight")
        urlopen.assert_called_once_with(msst_backend.MSST_MODEL_URL)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& 'D:/对比/RVCSVC-API/venv/Scripts/python.exe' -m unittest tests.test_msst_backend.MsstBackendTests.test_validate_assets_downloads_missing_model
```

Expected: FAIL because `validate_assets()` currently raises `FileNotFoundError` for a missing model.

- [ ] **Step 3: Write minimal implementation**

```python
MSST_MODEL_URL = "https://modelscope.cn/models/CCYellowStar/bs_roformer_karaoke_frazer_becruily/resolve/master/bs_roformer_karaoke_frazer_becruily.ckpt"

def ensure_model_downloaded(model_path: Path) -> Path:
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    temporary_path = model_path.with_suffix(model_path.suffix + ".part")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path.unlink(missing_ok=True)
    try:
        with urlopen(MSST_MODEL_URL) as response, temporary_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("download produced an empty file")
        temporary_path.replace(model_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download MSST model to {model_path}: {error}") from error
    return model_path
```

Call it from `validate_assets()` before the missing-model error branch, then preserve the YAML validation branch unchanged.

- [ ] **Step 4: Run focused tests to verify they pass**

Run:
```powershell
& 'D:/对比/RVCSVC-API/venv/Scripts/python.exe' -m unittest tests.test_msst_backend
```

Expected: all MSST backend tests pass.

- [ ] **Step 5: Verify error cleanup and skip behavior**

Add and run two focused tests: an existing non-empty model must not call `urlopen`; a thrown `urlopen` error must leave neither target nor `.part` file. Expected: PASS.

- [ ] **Step 6: Commit published code only**

```powershell
git -C 'D:/对比/RVCSVC-API-github' add msst_backend.py
git -C 'D:/对比/RVCSVC-API-github' commit -m 'Download MSST model on demand'
```

Do not stage `tests/` or local weights.

### Task 2: Verify integration and publish the branch

**Files:**
- Verify: `D:/对比/RVCSVC-API/app_rvc.py`
- Verify: `D:/对比/RVCSVC-API/app_svc.py`
- Modify: `D:/对比/RVCSVC-API-github/docs/superpowers/plans/2026-07-11-msst-model-download.md`

**Interfaces:**
- Consumes: `separate_msst_audio()` calls from both application entry points.
- Produces: a pushed PR branch containing only application code and documentation.

- [ ] **Step 1: Confirm both entry points still delegate to `separate_msst_audio()`**

Run:
```powershell
rg -n "separate_msst_audio|MSST_SELECTION" 'D:/对比/RVCSVC-API/app_rvc.py' 'D:/对比/RVCSVC-API/app_svc.py'
```

Expected: both files import the selection constant and call the backend adapter.

- [ ] **Step 2: Run the complete local test suite**

Run:
```powershell
& 'D:/对比/RVCSVC-API/venv/Scripts/python.exe' -m unittest discover -s 'D:/对比/RVCSVC-API/tests'
```

Expected: exit code 0 with no failures.

- [ ] **Step 3: Inspect and push the publish branch**

Run:
```powershell
git -C 'D:/对比/RVCSVC-API-github' status --short
git -C 'D:/对比/RVCSVC-API-github' push
```

Expected: only intended documentation/code changes are committed; push succeeds.
