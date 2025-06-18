import os
import pandas as pd
import numpy as np
from scipy.spatial import KDTree
import time # Thêm thư viện time để đo thời gian huấn luyện

# ======= THIẾT LẬP ĐƯỜNG DẪN =======
# !!! QUAN TRỌNG: Hãy đảm bảo đường dẫn này đúng với máy của bạn !!!
dump_folder = r"C:\Users\MAY02\Desktop\SLM_NeuralNetwork\LIGGGHTS\LIGGGHTS\LIGGGHTS\relax_simulation_Al\relax_Al_8line_1"
output_folder = "./output_SR_results"  # Thư mục lưu file CSV và biểu thức
output_file_csv = os.path.join(output_folder, "relax_Al_8line_1_processed.csv")  # Đường dẫn file CSV
output_expression_file = os.path.join(output_folder, "best_symbolic_expression_relax_Al_8line_1.txt") # File lưu biểu thức

# Tạo thư mục nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# ======= ĐỌC DỮ LIỆU TỪ FILE DUMP =======
all_data = []

def parse_dump_file(filepath):
    """Hàm đọc file dump & xử lý dữ liệu"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 9:
            #print(f"⚠️ File {filepath} quá ngắn, bỏ qua.")
            return [] 

        timestep_line = lines[1].strip()
        if not timestep_line.isdigit():
            #print(f"⚠️ Timestep không hợp lệ trong {filepath}: '{timestep_line}', bỏ qua.")
            return []
        timestep = int(timestep_line)

        num_atoms_line = lines[3].strip()
        if not num_atoms_line.isdigit():
            #print(f"⚠️ Số lượng nguyên tử không hợp lệ trong {filepath}: '{num_atoms_line}', bỏ qua.")
            return []
        num_atoms = int(num_atoms_line)

        if len(lines) < 9 + num_atoms:
            #print(f"⚠️ Không đủ dòng dữ liệu nguyên tử trong {filepath}. Dự kiến {num_atoms}, có {len(lines)-9} dòng. Bỏ qua file.")
            return []

        atom_data = lines[9:9 + num_atoms]
        data = []
        for i, line in enumerate(atom_data):
            cols = line.strip().split()
            if len(cols) < 6:
                #print(f"⚠️ Dòng {i+10} trong file {filepath} không đủ cột: {line.strip()}. Bỏ qua dòng này.")
                continue
            try:
                # Đảm bảo các cột cần thiết là id type x y z c_temp
                atom_id, atom_type, x, y, z, temp = map(float, cols[0:6]) # Lấy 6 cột đầu tiên
                if temp < 100: 
                    temp = 300.0
                data.append({
                    "Timestep": timestep,
                    "Atom_ID": int(atom_id),
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "Temperature": temp
                })
            except ValueError as ve:
                #print(f"⚠️ Lỗi chuyển đổi dữ liệu ở dòng {i+10} trong file {filepath}: {line.strip()} ({ve}). Bỏ qua dòng này.")
                continue
        return data
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file {filepath}")
        return []
    except Exception as e:
        print(f"❌ Lỗi xử lý file {filepath}: {e}")
        return []

# ======= XÁC ĐỊNH VỊ TRÍ LASER =======
def get_laser_position(timestep):
    omega = np.pi / 0.01
    dump_interval = 100000 

    initial_laser_x = 0.0105
    initial_laser_y = -0.01
    initial_laser_z = 0.01
    initial_laser_power = 1000

    laser_appearence_timestep = 100000 # Thời điểm laser xuất hiện lần đầu tiên
    scan_loop_start_timestep = 200000  # Thời điểm laser bắt đầu quét theo vòng lặp

    if timestep < laser_appearence_timestep:
         return None, None, None, 0 
    elif timestep < scan_loop_start_timestep: # Bao gồm timestep == laser_appearence_timestep
         return initial_laser_x, initial_laser_y, initial_laser_z, initial_laser_power
    
    current_relative_timestep = timestep - scan_loop_start_timestep 
    loop_step = current_relative_timestep // dump_interval
    
    num_points_per_scanline1 = 20
    num_scanlines1 = 3 
    total_steps_myloop1 = num_points_per_scanline1 * num_scanlines1 # 60 bước

    if loop_step < total_steps_myloop1:
        i = loop_step // num_points_per_scanline1 + 1
        k = loop_step % num_points_per_scanline1 + 1
        b = 0.001 * k - 0.01 
        a = 0.008 * np.sin(omega * b) + 0.0105 + 0.028 * (i - 1) + 0.0007 * k
        return a, b, initial_laser_z, initial_laser_power # Giữ laser_z và power cố định
    else: 
        loop_step_myloop2 = loop_step - total_steps_myloop1
        num_points_per_scanline2 = 20
        # Giả sử myloop2 tiếp tục cho các đường còn lại.
        # Nếu có tổng 8 đường, 3 ở loop1, vậy còn 5 ở loop2.
        # i_loop2 tính toán chỉ số đường quét hiện tại trong myloop2
        i_loop2 = loop_step_myloop2 // num_points_per_scanline2 + 1 
        j = loop_step_myloop2 % num_points_per_scanline2 + 1
        
        d = (-1) * 0.001 * j + 0.01 
        c = (-1) * 0.008 * np.sin(omega * d) + 0.0105 + 0.014 + (i_loop2 - 1) * 0.028 + 0.0007 * j
        return c, d, initial_laser_z, initial_laser_power # Giữ laser_z và power cố định

# ======= XỬ LÝ CÁC FILE DUMP =======
if not os.path.exists(dump_folder):
    print(f"❌ Thư mục dump '{dump_folder}' không tồn tại. Vui lòng kiểm tra lại đường dẫn.")
    exit()

dump_files = sorted([f for f in os.listdir(dump_folder) if f.startswith("coordtemp") and f.endswith(".dump")])

if not dump_files:
    print(f"⚠️ Không tìm thấy file dump nào trong thư mục '{dump_folder}'.")
    exit()

for filename in dump_files:
    filepath = os.path.join(dump_folder, filename)
    print(f"📂 Đang xử lý {filename}...")
    timestep_data = parse_dump_file(filepath)

    if not timestep_data:
        print(f"⚠️ Không có dữ liệu từ {filename} hoặc file bị lỗi, bỏ qua.")
        continue

    timestep = timestep_data[0]["Timestep"]
    laser_x, laser_y, laser_z, laser_power = get_laser_position(timestep)
    
    for atom in timestep_data:
        atom["Laser_X"] = laser_x
        atom["Laser_Y"] = laser_y
        atom["Laser_Z"] = laser_z
        atom["Laser_Power"] = laser_power # Đã là int từ get_laser_position
        if laser_x is not None and laser_y is not None and laser_z is not None:
            atom["Distance_To_Laser"] = np.sqrt(
                (atom["X"] - laser_x)**2 +
                (atom["Y"] - laser_y)**2 +
                (atom["Z"] - laser_z)**2
            )
        else:
            atom["Distance_To_Laser"] = np.nan
    all_data.extend(timestep_data)

if not all_data:
    print("❌ Không có dữ liệu nào được xử lý sau khi đọc các file dump. Kết thúc chương trình.")
    exit()

df = pd.DataFrame(all_data)
df = df.sort_values(["Timestep", "Atom_ID"]).reset_index(drop=True)

# ======= TÍNH TOÁN NHIỆT ĐỘ HÀNG XÓM =======
df["Prev_Timestep_Temp"] = df.groupby("Atom_ID")["Temperature"].shift(1)

cutoff_distance = 0.002 
k_neighbors_for_kdtree = 30 

def compute_neighbor_stats(df_timestep_current, df_timestep_previous):
    # df_timestep_current: DataFrame của timestep hiện tại (t)
    # df_timestep_previous: DataFrame của timestep trước đó (t-1)
    if df_timestep_previous is None or df_timestep_previous.empty:
        df_timestep_current["Max_Neighbor_Temp"] = np.nan
        df_timestep_current["Min_Neighbor_Temp"] = np.nan
        df_timestep_current["Avg_Neighbor_Temp"] = np.nan
        df_timestep_current["Closest_Neighbor_Temp"] = np.nan
        df_timestep_current["Neighbor_Count"] = 0 
        return df_timestep_current

    prev_positions = df_timestep_previous[["X", "Y", "Z"]].values
    # Sử dụng nhiệt độ 'Temperature' của timestep trước làm nhiệt độ hàng xóm
    prev_temps = df_timestep_previous["Temperature"].values 

    if len(prev_positions) == 0:
        df_timestep_current["Max_Neighbor_Temp"] = np.nan
        df_timestep_current["Min_Neighbor_Temp"] = np.nan
        df_timestep_current["Avg_Neighbor_Temp"] = np.nan
        df_timestep_current["Closest_Neighbor_Temp"] = np.nan
        df_timestep_current["Neighbor_Count"] = 0
        return df_timestep_current
        
    tree = KDTree(prev_positions)
    current_positions = df_timestep_current[["X", "Y", "Z"]].values
    
    distances, indices = tree.query(
        current_positions, 
        k=min(k_neighbors_for_kdtree, len(prev_positions)), # k không thể lớn hơn số điểm trong tree
        distance_upper_bound=cutoff_distance
    )
    
    max_temps, min_temps, avg_temps, closest_temps_list, neighbor_counts = [], [], [], [], []

    for i in range(len(current_positions)): 
        dist_row = distances[i]
        idx_row = indices[i]
        
        # Xử lý trường hợp distances/indices không phải là array (khi k=1 và chỉ có 1 hàng xóm)
        if not isinstance(dist_row, np.ndarray):
            dist_row = np.array([dist_row])
            idx_row = np.array([idx_row])

        # idx_row có thể chứa len(prev_positions) nếu không đủ k hàng xóm trong cutoff
        # hoặc nếu k > số lượng thực tế của prev_positions.
        # Loại bỏ các chỉ số không hợp lệ ( >= len(prev_positions) )
        valid_mask = (dist_row < cutoff_distance) & (idx_row < len(prev_positions))
        valid_neighbor_indices_in_prev_step = idx_row[valid_mask]
        valid_distances = dist_row[valid_mask]

        neighbor_counts.append(len(valid_neighbor_indices_in_prev_step))

        if len(valid_neighbor_indices_in_prev_step) > 0:
            neighbor_actual_temps = prev_temps[valid_neighbor_indices_in_prev_step]
            max_temps.append(np.max(neighbor_actual_temps))
            min_temps.append(np.min(neighbor_actual_temps))
            avg_temps.append(np.mean(neighbor_actual_temps))
            closest_neighbor_local_idx = np.argmin(valid_distances)
            closest_neighbor_global_idx = valid_neighbor_indices_in_prev_step[closest_neighbor_local_idx]
            closest_temps_list.append(prev_temps[closest_neighbor_global_idx])
        else:
            max_temps.append(np.nan)
            min_temps.append(np.nan)
            avg_temps.append(np.nan)
            closest_temps_list.append(np.nan)
            
    df_timestep_current["Max_Neighbor_Temp"] = max_temps
    df_timestep_current["Min_Neighbor_Temp"] = min_temps
    df_timestep_current["Avg_Neighbor_Temp"] = avg_temps
    df_timestep_current["Closest_Neighbor_Temp"] = closest_temps_list
    df_timestep_current["Neighbor_Count"] = neighbor_counts 
    
    return df_timestep_current

timesteps_unique = sorted(df["Timestep"].unique())
processed_dfs = []
df_prev_timestep_for_neighbors = None 

for ts_val in timesteps_unique:
    print(f"⏳ Tính toán hàng xóm cho Timestep: {ts_val}")
    df_current_timestep = df[df["Timestep"] == ts_val].copy() # Dữ liệu timestep t
    
    # df_prev_timestep_for_neighbors là dữ liệu của timestep t-1
    df_current_timestep_with_neighbors = compute_neighbor_stats(df_current_timestep, df_prev_timestep_for_neighbors)
    processed_dfs.append(df_current_timestep_with_neighbors)

    # Cập nhật df_prev_timestep_for_neighbors cho vòng lặp tiếp theo
    # Chỉ cần các cột cần thiết để xây KDTree và lấy nhiệt độ
    df_prev_timestep_for_neighbors = df[df["Timestep"] == ts_val][["X", "Y", "Z", "Temperature"]].copy()

if processed_dfs:
    df = pd.concat(processed_dfs, ignore_index=True)
    df = df.sort_values(["Timestep", "Atom_ID"]).reset_index(drop=True)
else:
    print("❌ Không có dữ liệu sau khi xử lý hàng xóm.")
    exit()
    
# ======= LƯU FILE CSV =======
df.to_csv(output_file_csv, index=False)
print(f"✅ Dữ liệu đã được xử lý và lưu vào {output_file_csv}")

# ======= PHÂN TÍCH FEATURE CHO SYMBOLIC REGRESSION =======
print("\n" + "="*30)
print(" BẮT ĐẦU PHÂN TÍCH VÀ CHUẨN BỊ CHO SYMBOLIC REGRESSION ")
print("="*30 + "\n")

df_sr = df.copy() # Sử dụng df đã xử lý ở trên

print("\n📊 Thông tin DataFrame cho Symbolic Regression (df_sr):")
df_sr.info(verbose=True, show_counts=True)
print("\n📈 Thống kê mô tả các cột số:")
print(df_sr.describe())

target_column = "Temperature"
potential_features = [
    "Laser_X", "Laser_Y", "Laser_Z", 
    "Laser_Power", "Distance_To_Laser", "Prev_Timestep_Temp",
    "Max_Neighbor_Temp", "Min_Neighbor_Temp", "Avg_Neighbor_Temp", 
    "Closest_Neighbor_Temp", "Neighbor_Count",
    "X", "Y", "Z" 
]
selected_features = [col for col in potential_features if col in df_sr.columns]
print(f"\n🎯 Target column: {target_column}")
print(f"🧬 Selected features ({len(selected_features)}): {selected_features}")

print(f"\nSố dòng ban đầu trong df_sr: {len(df_sr)}")
cols_to_check_for_nan = [target_column] + selected_features
df_sr_cleaned = df_sr.dropna(subset=cols_to_check_for_nan).copy()
print(f"Số dòng sau khi loại bỏ NaN ở target và các feature đã chọn: {len(df_sr_cleaned)}")

if df_sr_cleaned.empty:
    print("\n❌ Không còn dữ liệu sau khi làm sạch NaN. Không thể tiến hành Symbolic Regression.")
    exit() 
else:
    X_sr = df_sr_cleaned[selected_features]
    y_sr = df_sr_cleaned[target_column]
    print("\n🔬 Dữ liệu đã sẵn sàng cho Symbolic Regression:")
    print(f"Hình dạng của X_sr (features): {X_sr.shape}")
    print(f"Hình dạng của y_sr (target): {y_sr.shape}")

# ======= ÁP DỤNG THUẬT TOÁN SYMBOLIC REGRESSION VỚI GPLEARN =======
print("\n" + "="*30)
print(" BẮT ĐẦU HUẤN LUYỆN SYMBOLIC REGRESSION VỚI GPLEARN ")
print("="*30 + "\n")

if 'X_sr' in locals() and 'y_sr' in locals() and not X_sr.empty:
    try:
        from gplearn.genetic import SymbolicRegressor
        # from gplearn.functions import make_function # Nếu cần hàm tùy chỉnh

        X_train_np = X_sr.values
        y_train_np = y_sr.values
        feature_names_sr = X_sr.columns.tolist() # Đổi tên để tránh trùng với feature_names trong một số hàm khác

        function_set_sr = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos'] # Đổi tên
        
        print(f"🧬 Sử dụng các features: {feature_names_sr}")
        print(f"🛠️ Bộ hàm được sử dụng: {function_set_sr}")
        print(f"⏳ Bắt đầu quá trình huấn luyện Symbolic Regressor...")
        print("Lưu ý: Quá trình này có thể mất nhiều thời gian.")
        
        sr_start_time = time.time() # Đổi tên biến thời gian

        # Khởi tạo SymbolicRegressor
        # Cân nhắc điều chỉnh các tham số này
        est_gp = SymbolicRegressor(population_size=1000, 
                                   generations=20,       
                                   stopping_criteria=0.01, 
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   max_samples=0.9, verbose=1,
                                   feature_names=feature_names_sr, # Sử dụng tên đã đổi
                                   function_set=function_set_sr,   # Sử dụng tên đã đổi
                                   metric='mean absolute error', # Có thể thay đổi, ví dụ 'mse'
                                   parsimony_coefficient=0.005, 
                                   random_state=42,
                                   n_jobs=-1 
                                  )
        est_gp.fit(X_train_np, y_train_np)
        sr_end_time = time.time() # Đổi tên biến thời gian
        sr_training_time = sr_end_time - sr_start_time # Đổi tên biến thời gian

        print(f"\n✅ Huấn luyện hoàn tất trong {sr_training_time:.2f} giây.")

        if hasattr(est_gp, '_program') and est_gp._program is not None:
            print("\n🏆 Biểu thức Symbolic Regression tốt nhất tìm được:")
            best_expression_str = str(est_gp._program)
            print(best_expression_str)
            # Sử dụng est_gp.metric để lấy tên metric đã dùng khi khởi tạo
            print(f"\n📈 Fitness (Sử dụng metric: '{est_gp.metric}'): {est_gp._program.fitness_}")

            try:
                with open(output_expression_file, 'w', encoding='utf-8') as f: # Thêm encoding
                    f.write("Biểu thức Symbolic Regression tốt nhất tìm được:\n")
                    f.write(best_expression_str + "\n\n")
                    f.write(f"Fitness (Sử dụng metric: '{est_gp.metric}'): {est_gp._program.fitness_}\n")
                    f.write(f"Huấn luyện trong: {sr_training_time:.2f} giây\n")
                    f.write(f"Các tham số gplearn chính:\n")
                    f.write(f"  population_size: {est_gp.population_size}\n")
                    f.write(f"  generations: {est_gp.generations}\n")
                    f.write(f"  parsimony_coefficient: {est_gp.parsimony_coefficient}\n")
                    f.write(f"  stopping_criteria: {est_gp.stopping_criteria}\n")
                    f.write(f"  metric: {est_gp.metric}\n")
                    f.write(f"  function_set: {est_gp.function_set}\n") # est_gp.function_set là một list các đối tượng hàm
                    f.write(f"  feature_names: {est_gp.feature_names}\n")
                print(f"\n📄 Biểu thức và thông tin liên quan đã được lưu vào file: {output_expression_file}")
            except Exception as e_file:
                print(f"❌ Đã xảy ra lỗi khi cố gắng lưu biểu thức ra file: {e_file}")
        else:
            print("\n⚠️ Không tìm thấy chương trình tối ưu (_program) trong đối tượng SymbolicRegressor để lưu.")

    except ImportError:
        print("⚠️ Thư viện gplearn chưa được cài đặt. Vui lòng cài đặt bằng: pip install gplearn")
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình Symbolic Regression: {e}")
else:
    print("\n⚠️ Không tìm thấy dữ liệu X_sr hoặc y_sr, hoặc dữ liệu rỗng. Bỏ qua Symbolic Regression.")

# ======= THAM KHẢO: SỬ DỤNG PYSR (NẾU MUỐN THỬ) =======
# print("\n" + "="*30)
# print(" THAM KHẢO: SYMBOLIC REGRESSION VỚI PYSR ")
# print("="*30 + "\n")
# print("Lưu ý: Để chạy PySR, bạn cần cài đặt PySR (pip install pysr) và Julia.")
# ... (code PySR đã comment có thể giữ lại ở đây nếu muốn) ...

print("\n🏁 Chương trình đã hoàn thành.")