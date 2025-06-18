import math

# --- Các giá trị từ Verilog ---
p = 113680897410347
q = 7999808077935876437321
msg_in_hex = "0000000000000000000000000000000000262d806a3e18f03ab37b2857e7e149"
msg_in = int(msg_in_hex, 16)

# --- Giả định số mũ công khai (e) ---
# Đây là giá trị phổ biến nhất, BẠN CẦN KIỂM TRA LẠI TRONG MÃ VERILOG CỦA MÌNH
e = 65537

# --- Tính toán các tham số RSA ---

# 1. Tính N (Modulus)
N = p * q

# 2. Tính Phi(N) (Euler's Totient Function)
phi_n = (p - 1) * (q - 1)

# 3. Tính d (Số mũ bí mật)
# Sử dụng hàm pow(base, exp, mod) của Python 3.8+ để tính nghịch đảo modular
# pow(e, -1, phi_n) tương đương với d * e = 1 (mod phi_n)
try:
    d = pow(e, -1, phi_n)
except ValueError:
    print("Không thể tính d. Kiểm tra lại p, q, và e.")
    exit()

# --- Thực hiện Mã hóa và Giải mã ---

# 1. Mã hóa (Encrypt)
# C = M^e mod N
ciphertext = pow(msg_in, e, N)

# 2. Giải mã (Decrypt) - Để kiểm tra lại
# M_dec = C^d mod N
decrypted_msg = pow(ciphertext, d, N)

# --- In kết quả ---
print(f"p = {p}")
print(f"q = {q}")
print("-" * 30)
print(f"N (p * q) = {N}")
print(f"Phi(N)    = {phi_n}")
print("-" * 30)
print(f"e (Public Key) = {e}")
print(f"d (Private Key)= {d}")
print("-" * 30)
print(f"Message (M)    = {msg_in}")
print(f"Message (hex)  = {hex(msg_in)}")
print("-" * 30)
print(f"Ciphertext (C) = {ciphertext}")
print(f"Ciphertext (hex)= {hex(ciphertext)}") # <-- ĐÂY LÀ GIÁ TRỊ CẦN SO SÁNH VỚI msg_out
print("-" * 30)
print(f"Decrypted (M') = {decrypted_msg}")
print(f"Decrypted (hex)= {hex(decrypted_msg)}")
print("-" * 30)

# --- Kiểm tra ---
if msg_in == decrypted_msg:
    print("✅ Kiểm tra thành công: Tin nhắn giải mã khớp với tin nhắn gốc.")
else:
    print("❌ Lỗi: Tin nhắn giải mã KHÔNG khớp với tin nhắn gốc.")