"""网易云歌曲下载：多源依次尝试，取第一个真正拿得到的音频。

原来的两条路都坏了：
- 主源 biliplayer.91vrchat.com 对大量歌曲直接 404；
- 备用源 api.vkeys.cn 现在统一返回 {"code": 500, "data": []}，
  旧代码写死 ["data"]["url"]，于是抛
  TypeError: list indices must be integers or slices, not str
  ——这就是界面上看到的 "所有下载源均失败: list indices must be integers or slices, not str"。

现在的做法：按「拿 mp3 的成本」从低到高排队，每个源都校验返回的确实是音频
（文件头魔数 + 体积下限，挡掉错误页和试听片段），不行就换下一个；
全挂时把每个源各自的原因拼进异常，下次一眼能看出是哪家失效了。
"""

import json
import re

import requests

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_MIN_BYTES = 1_000_000  # 小于 1MB 的多半是错误页或试听片段
_AUDIO_MAGIC = (
    b"ID3",        # mp3
    b"\xff\xfb",   # mp3 frame sync
    b"\xff\xfa",
    b"\xff\xf3",
    b"\xff\xf2",
    b"fLaC",       # flac
    b"OggS",       # ogg/opus
    b"RIFF",       # wav
)


_MIN_BYTES = 1_000_000  # 超过这个大小直接放行
_MIN_SECONDS = 45       # 小于这个大小才去量时长；短于此时长视为试听片段
_MIN_BYTES_ANY = 64 * 1024  # 小于这个大小基本是错误页，连量都不用量


def _measure_seconds(content: bytes) -> float:
    """用 pydub(ffmpeg) 量一下音频时长；失败返回 0。"""
    try:
        from io import BytesIO

        from pydub import AudioSegment

        return len(AudioSegment.from_file(BytesIO(content))) / 1000.0
    except Exception:
        return 0.0


def _looks_like_audio(content: bytes) -> bool:
    """判断是不是可用音频。

    大文件直接放行；小文件（<1MB）可能是短歌、也可能是 30 秒试听片段，
    所以量一下时长再决定，避免把合法的短歌误判掉。
    """
    if len(content) < _MIN_BYTES_ANY:
        return False
    is_audio = content[:4].startswith(_AUDIO_MAGIC) or content[4:8] == b"ftyp"
    if not is_audio:
        return False
    if len(content) >= _MIN_BYTES:
        return True
    seconds = _measure_seconds(content)
    if seconds >= _MIN_SECONDS:
        _log(f"小文件但时长 {seconds:.0f} 秒，按短音频放行（{len(content) / 1024:.0f} KB）")
        return True
    _log(f"只包含 {seconds:.0f} 秒音频（{len(content) / 1024:.0f} KB），判为试听片段")
    return False


def _log(message: str) -> None:
    """控制台是 GBK 时，个别第三方返回的字符会让 print 直接抛 UnicodeEncodeError。"""
    try:
        print(message)
    except UnicodeEncodeError:
        print(message.encode("gbk", "replace").decode("gbk", "replace"))


def _get_bytes(url: str, timeout: int = 60, headers: dict | None = None) -> bytes:
    merged = {"User-Agent": _UA}
    if headers:
        merged.update(headers)
    response = requests.get(url, headers=merged, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    return response.content


def _get_json(url: str, timeout: int = 30):
    response = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout, allow_redirects=True)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    return response.json()


def _pick_url(payload) -> str:
    """从各家形状不一的 JSON 里挑一个音频直链（data 可能是 dict 也可能是 list）。"""
    if not payload:
        return ""
    if isinstance(payload, str):
        return payload if payload.startswith("http") else ""
    if isinstance(payload, list):
        for item in payload:
            found = _pick_url(item)
            if found:
                return found
        return ""
    if isinstance(payload, dict):
        for key in ("url", "music_url", "src", "play_url"):
            value = payload.get(key)
            if isinstance(value, str) and value.startswith("http"):
                return value
        for key in ("data", "result", "song", "songs"):
            if key in payload:
                found = _pick_url(payload[key])
                if found:
                    return found
    return ""


# ---------------- 各个源 ----------------

def _source_91vrchat(song_id: str) -> bytes:
    return _get_bytes(
        f"https://biliplayer.91vrchat.com/player/?url=https://music.163.com/song?id={song_id}"
    )


def _source_gdstudio(song_id: str) -> bytes:
    payload = _get_json(
        f"https://music-api.gdstudio.xyz/api.php?types=url&source=netease&id={song_id}&br=320"
    )
    link = _pick_url(payload)
    if not link:
        raise RuntimeError(
            "接口没给直链（多为 VIP/版权曲）: "
            + json.dumps(payload, ensure_ascii=False)[:120]
        )
    return _get_bytes(link, timeout=180)


def _source_meting_qijieya(song_id: str) -> bytes:
    """qijieya 的 meting 实例：实测能拿到完整音轨（inj46ahow 同曲只给 ~480KB 试听）。"""
    return _get_bytes(
        f"https://api.qijieya.cn/meting/?server=netease&type=url&id={song_id}", timeout=180
    )


def _source_meting(song_id: str) -> bytes:
    return _get_bytes(f"https://api.injahow.cn/meting/?server=netease&type=url&id={song_id}")


def _source_outer(song_id: str) -> bytes:
    return _get_bytes(f"https://music.163.com/song/media/outer/url?id={song_id}.mp3")


def _source_vkeys(song_id: str) -> bytes:
    payload = _get_json(f"https://api.vkeys.cn/v2/music/netease?id={song_id}")
    link = _pick_url(payload)
    if not link:
        raise RuntimeError("接口没给直链: " + json.dumps(payload, ensure_ascii=False)[:120])
    return _get_bytes(link)


_SOURCES = (
    ("91vrchat", _source_91vrchat),
    ("gdstudio", _source_gdstudio),
    ("meting-qijieya", _source_meting_qijieya),
    ("meting", _source_meting),
    ("outer", _source_outer),
    ("vkeys", _source_vkeys),
)

# ---------------- B 站 ----------------

_BILI_HEADERS = {
    "User-Agent": _UA,
    "Referer": "https://www.bilibili.com/",
}


def _bili_get(url: str, params: dict, timeout: int = 30):
    response = requests.get(url, params=params, headers=_BILI_HEADERS, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}")
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"接口 code={payload.get('code')} {str(payload.get('message'))[:60]}")
    return payload.get("data") or {}


def _bili_video_info(bvid: str) -> dict:
    try:
        data = _bili_get("https://api.bilibili.com/x/web-interface/view", {"bvid": bvid})
    except RuntimeError as error:
        raise RuntimeError(f"{error}（BV 号区分大小写，请从视频页复制完整 BV 号）") from error
    if not data.get("cid"):
        raise RuntimeError("视频信息里没有 cid")
    return data


def _bili_download_audio(bvid: str) -> bytes:
    """优先级：dash 最高带宽音轨(m4a) → durl 单段(mp4)。"""
    info = _bili_video_info(bvid)
    real_bvid, cid = info["bvid"], int(info["cid"])
    problems = []

    try:
        data = _bili_get(
            "https://api.bilibili.com/x/player/playurl",
            {"bvid": real_bvid, "cid": cid, "fnval": 16, "fnver": 0, "fourk": 1},
        )
        audios = (data.get("dash") or {}).get("audio") or []
        if not audios:
            raise RuntimeError("没有 dash 音频流（可能该视频需要登录）")
        best = max(audios, key=lambda item: item.get("bandwidth") or 0)
        url = best.get("baseUrl") or best.get("base_url")
        if not url:
            raise RuntimeError("dash 音频流没有地址")
        content = _get_bytes(url, timeout=180, headers=_BILI_HEADERS)
        if _looks_like_audio(content):
            print(f"B站 dash 音轨命中（{best.get('bandwidth')} bps，{len(content) / 1024 / 1024:.1f} MB）")
            return content
        problems.append(f"dash: 返回内容不是音频（{len(content)} 字节）")
    except Exception as error:
        problems.append(f"dash: {type(error).__name__} {error}")
        print(f"B站 dash 下载失败: {error}")

    try:
        data = _bili_get(
            "https://api.bilibili.com/x/player/playurl",
            {"bvid": real_bvid, "cid": cid, "fnval": 0, "fnver": 0, "fourk": 0},
        )
        durl = data.get("durl") or []
        if not durl:
            raise RuntimeError("没有 durl 分片")
        content = _get_bytes(durl[0]["url"], timeout=180, headers=_BILI_HEADERS)
        if _looks_like_audio(content):
            print(f"B站 durl 命中（{len(content) / 1024 / 1024:.1f} MB）")
            return content
        problems.append(f"durl: 返回内容不是音频（{len(content)} 字节）")
    except Exception as error:
        problems.append(f"durl: {type(error).__name__} {error}")
        print(f"B站 durl 下载失败: {error}")

    raise RuntimeError("B站音频下载失败 -> " + " | ".join(problems))


_BV_PATTERN = re.compile(r"BV[0-9A-Za-z]{10}", re.IGNORECASE)
_AV_PATTERN = re.compile(r"(?:^|[^0-9A-Za-z])av(\d+)", re.IGNORECASE)


def extract_bilibili_id(text) -> str | None:
    """从 BV 号 / av 号 / B 站视频链接里抠出视频标识，认不出来返回 None。"""
    text = str(text or "").strip()
    match = _BV_PATTERN.search(text)
    if match:
        return "BV" + match.group(0)[2:]
    match = _AV_PATTERN.search(text)
    if match:
        return "av" + match.group(1)
    return None


def _is_bilibili_id(identifier: str) -> bool:
    lowered = identifier.lower()
    return lowered.startswith("bv") or lowered.startswith("av")


def fetch_song_audio(song_id) -> bytes:
    """入口分派：纯数字走网易云多源，BV/av 走 B 站。"""
    song_id = str(song_id).strip()

    if _is_bilibili_id(song_id):
        print(f"开始下载 B 站视频音频 {song_id}")
        return _bili_download_audio(song_id)

    if not song_id.isdigit():
        raise ValueError(f"无法识别的歌曲标识（既不是网易云 id 也不是 BV 号）: {song_id!r}")

    print(f"开始下载歌曲 {song_id}")
    problems = []
    for name, fetcher in _SOURCES:
        try:
            content = fetcher(song_id)
        except Exception as error:
            problems.append(f"{name}: {type(error).__name__} {error}")
            _log(f"下载源 {name} 失败: {error}")
            continue
        if not _looks_like_audio(content):
            problems.append(f"{name}: 返回的不是音频（{len(content)} 字节）")
            _log(f"下载源 {name} 返回的不是音频（{len(content)} 字节），换下一个")
            continue
        _log(f"下载源 {name} 命中，{len(content) / 1024 / 1024:.1f} MB")
        return content

    raise RuntimeError("所有下载源均失败 -> " + " | ".join(problems))
