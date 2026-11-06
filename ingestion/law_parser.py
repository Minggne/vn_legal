"""Phân tích văn bản luật thành Điều/Mục/Đoạn rồi tạo chunk có đường dẫn ngữ cảnh."""
import re

RE_CHUONG = re.compile(r'^Chương\s+([IVXLC]+|\d+)\b\.?\s*(.*)$', re.I)
RE_MUC    = re.compile(r'^Mục\s+(\d+)\b\.?\s*(.*)$', re.I)
RE_DIEU   = re.compile(r'^Điều\s+(\d+[a-zA-Z]?)\s*[.:]\s*(.*)$', re.I)
RE_DIEU_TRAN = re.compile(r'^Điều\s+(\d+[a-zA-Z]?)\s*$', re.I)     # "Điều 1" đứng một mình (văn bản cũ)
RE_KHOAN  = re.compile(r'^(\d+)\.\s+(.+)$')
RE_LAMA   = re.compile(r'^([IVX]{1,6})\s*[.)]\s*(.*)$')            # "I. QUY ĐỊNH CHUNG"
RE_SO     = re.compile(r'^(\d{1,2})\s*[.)]\s+(.+)$')               # "1. Nội dung"
RE_DIEM   = re.compile(r'^[a-zđ]\)\s')                              # "a) ...", "đ) ..."
MIN_CHARS = 300
MAX_CHUNK = 3000       # chunk tìm kiếm dài hơn mức này sẽ được chia nhỏ (BGE-M3 chỉ đọc ~1024 token đầu)
PIECE = 2200           # kích thước tối đa mỗi phần sau khi chia


# =========================================================
# STRUCTURE PARSING
# =========================================================

def _la_dieu(line):
    return bool(RE_DIEU.match(line) or RE_DIEU_TRAN.match(line))


def _giong_tieu_de(line):
    """Dòng ngắn, không phải câu hoàn chỉnh: có thể là tên Điều nằm ở dòng sau 'Điều N'."""
    return (len(line) <= 150 and not line.endswith((".", ";", ":", ","))
            and not RE_KHOAN.match(line) and not RE_CHUONG.match(line)
            and not RE_MUC.match(line) and not _la_dieu(line))


def parse_dieu(text):
    """Trả về danh sách Điều: {so, ten, chuong, lines, nhan}."""
    dieus, cur = [], None
    chuong, pending, cho_ten = None, False, False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := RE_CHUONG.match(line):
            chuong = f"Chương {m[1].upper()}"
            if m[2].strip():
                chuong += f" {m[2].strip()}"
                pending = False
            else:
                pending = True          # tên chương nằm ở dòng kế tiếp
            cho_ten = False
            continue
        if pending and not _la_dieu(line):
            chuong += f" {line}"
            pending = False
            continue
        pending = False
        if RE_MUC.match(line):
            cho_ten = False
            continue
        so = ten = None
        if m := RE_DIEU.match(line):
            so, ten = m[1], m[2].strip()
        elif m := RE_DIEU_TRAN.match(line):
            so, ten = m[1], ""
        if so is not None:
            cur = {"so": so, "ten": ten, "chuong": chuong, "lines": [], "nhan": "Điều"}
            dieus.append(cur)
            cho_ten = not ten
            continue
        if cur is not None:
            if cho_ten and _giong_tieu_de(line):
                cur["ten"] = line               # tên Điều nằm ở dòng kế tiếp
                cho_ten = False
                continue
            cho_ten = False
            cur["lines"].append(line)
    return dieus


def _tach(lines, regex, nhan="Mục"):
    """Tách theo dòng khớp regex; phần trước dòng khớp đầu tiên (quốc hiệu, căn cứ...) bị bỏ.
    Mục La Mã: phần chữ sau số (nếu ngắn) làm tên mục. Mục đánh số: giữ nguyên cả dòng làm nội dung."""
    secs, cur, dem, cho_ten = [], None, {}, False
    for ln in lines:
        m = regex.match(ln)
        if m:
            so = m[1]
            dem[so] = dem.get(so, 0) + 1
            if dem[so] > 1:                     # số lặp lại (đánh số lại từ đầu): thêm hậu tố cho duy nhất
                so = f"{so}-{dem[so]}"
            ten = m[2].strip() if m.lastindex and m.lastindex >= 2 else ""
            cur = {"so": so, "ten": "", "chuong": None, "lines": [], "nhan": nhan}
            if regex is RE_LAMA and 0 < len(ten) <= 120:
                cur["ten"] = ten
            elif ten:
                cur["lines"].append(ln)         # tiêu đề quá dài hoặc mục đánh số: giữ cả dòng làm nội dung
            cho_ten = regex is RE_LAMA and not ten
            secs.append(cur)
        elif cur is not None:
            if cho_ten and _giong_tieu_de(ln):
                cur["ten"] = ln                 # "I." đứng một mình, tên mục ở dòng sau
                cho_ten = False
                continue
            cho_ten = False
            cur["lines"].append(ln)
    return secs


def _doan(lines, cua_so=1800):
    secs, buf, n = [], [], 0
    for ln in lines:
        buf.append(ln)
        n += len(ln)
        if n >= cua_so:
            secs.append({"so": str(len(secs) + 1), "ten": "", "chuong": None, "lines": buf, "nhan": "Đoạn"})
            buf, n = [], 0
    if buf:
        secs.append({"so": str(len(secs) + 1), "ten": "", "chuong": None, "lines": buf, "nhan": "Đoạn"})
    return secs


def parse_muc(text):
    """Dự phòng cho văn bản không chia theo Điều. Trả về (danh sách phần, kiểu)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    secs = _tach(lines, RE_LAMA)
    if len(secs) >= 2:
        return secs, "muc"
    secs = _tach(lines, RE_SO)
    if len(secs) >= 3:
        return secs, "muc"
    return _doan(lines), "doan"


def parse_van_ban(text):
    if len(text) < MIN_CHARS:
        return [], "ngan"
    dieus = parse_dieu(text)
    if len(dieus) == 1 and len(text) > 4000:
        # Chỉ một "Điều" nhưng văn bản dài và có từ 3 mục La Mã trở lên: nhiều khả năng "Điều" là dòng lẻ
        # (dẫn chiếu bị ngắt dòng). Chỉ xét mục La Mã, KHÔNG xét mục đánh số (1., 2.) vì một Điều dài
        # có nhiều Khoản cũng có dạng đó.
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        secs = _tach(lines, RE_LAMA)
        if len(secs) >= 3:
            return secs, "muc"
    if dieus:
        return dieus, "dieu"
    return parse_muc(text)


# =========================================================
# CHUNKING
# =========================================================

def split_khoan(lines):
    parts, k, buf = [], None, []
    for ln in lines:
        if m := RE_KHOAN.match(ln):
            if buf:
                parts.append((k, "\n".join(buf)))
            k, buf = m[1], [ln]
        else:
            buf.append(ln)
    if buf:
        parts.append((k, "\n".join(buf)))
    return parts


def _cat_dong(line, toi_da):
    """Cắt một dòng quá dài thành các mẩu <= toi_da ký tự, ưu tiên ranh giới câu."""
    if len(line) <= toi_da:
        return [line]
    out, buf = [], ""
    for cau in re.split(r'(?<=[.;])\s+', line):
        while len(cau) > toi_da:                         # câu quá dài: cắt cứng
            if buf:
                out.append(buf)
                buf = ""
            out.append(cau[:toi_da])
            cau = cau[toi_da:]
        if buf and len(buf) + 1 + len(cau) > toi_da:
            out.append(buf)
            buf = cau
        else:
            buf = f"{buf} {cau}".strip() if buf else cau
    if buf:
        out.append(buf)
    return out


def chia_nho(text, toi_da=PIECE):
    """Chia một đoạn dài thành các phần <= toi_da ký tự, ưu tiên cắt ở ranh giới điểm a), b)...
    Các phần sau lặp lại dòng dẫn đầu (ví dụ '1. Nội dung chi:') để không mất ngữ cảnh."""
    if len(text) <= MAX_CHUNK:
        return [text]
    lines = text.split("\n")
    lead = lines[0][:200]
    blocks = []
    for ln in lines:
        if not blocks or RE_DIEM.match(ln):
            blocks.append([ln])
        else:
            blocks[-1].append(ln)
    mau = []                                             # các khối đã đảm bảo <= toi_da
    for b in blocks:
        t = "\n".join(b)
        if len(t) <= toi_da:
            mau.append(t)
            continue
        buf = ""
        for ln in b:
            for m in _cat_dong(ln, toi_da):
                if buf and len(buf) + 1 + len(m) > toi_da:
                    mau.append(buf)
                    buf = m
                else:
                    buf = f"{buf}\n{m}" if buf else m
        if buf:
            mau.append(buf)
    pieces, buf = [], ""
    for m in mau:
        if buf and len(buf) + 1 + len(m) > toi_da:
            pieces.append(buf)
            buf = m
        else:
            buf = f"{buf}\n{m}" if buf else m
    if buf:
        pieces.append(buf)
    return [pc if i == 0 or pc.startswith(lead) else f"{lead}\n{pc}" for i, pc in enumerate(pieces)]


def build_chunks(ten_vb, so_hieu, dieus, max_chars=2500):
    """Phần ngắn -> 1 chunk tìm kiếm. Phần dài -> phần làm cha (không tìm kiếm), tìm theo từng Khoản."""
    out = []
    for d in dieus:
        nhan = d.get("nhan", "Điều")
        path = [f"{ten_vb} ({so_hieu})"]
        if d["chuong"]:
            path.append(d["chuong"])
        path.append(f"{nhan} {d['so']}. {d['ten']}".strip())
        full = f"{nhan} {d['so']}. {d['ten']}\n".replace(". \n", ".\n") + "\n".join(d["lines"])
        short = len(full) <= max_chars
        idx = len(out)
        out.append(dict(cap="dieu", nhan=nhan, dieu=d["so"], khoan=None, duong_dan=" > ".join(path),
                        noi_dung=full, parent=None, tim_kiem=short))
        if not short:
            for k, text in split_khoan(d["lines"]):
                p = path + ([f"Khoản {k}"] if k else [])
                pieces = chia_nho(text)
                for j, pc in enumerate(pieces, 1):
                    pp = p + ([f"Phần {j}/{len(pieces)}"] if len(pieces) > 1 else [])
                    out.append(dict(cap="khoan", nhan=nhan, dieu=d["so"], khoan=k, duong_dan=" > ".join(pp),
                                    noi_dung=pc, parent=idx, tim_kiem=True))
    for c in out:
        c["tien_to"] = c["duong_dan"] + "\n" + c["noi_dung"]
    return out
