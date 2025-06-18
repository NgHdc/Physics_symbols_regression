import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def simulate_slm_layer_fdm_and_plot_heatmaps(
    nx=60, nz_layer=3, nz_base=10, n_layers=3, # Giảm số lớp để test nhanh
    alpha=2e-6,  # Giảm alpha để nhiệt ít lan tỏa hơn (m^2/s)
    dx=5e-5,     # Kích thước lưới x (m)
    dz=5e-5,     # Kích thước lưới z (m)
    dt=2e-5,     # Giảm dt để ổn định với alpha mới (s)
    laser_speed=0.1, # Giảm tốc độ laser để tăng tương tác (m/s)
    laser_radius=2e-4, # Bán kính chùm tia (m) - có thể giảm để tập trung nhiệt
    target_laser_temp=1000.0, # Nhiệt độ mục tiêu dưới laser (K)
    t_scan_layer=None,
    t_cool_layer=0.05,  # Thời gian nguội 1 lớp (s)
    T_ambient=300.0,   # Nhiệt độ môi trường (K)
    plot_layers=[1, 2, 3],
    colormap='inferno', # Thử colormap 'inferno' hoặc 'magma'
    plot_during_scan_layer=1 # Lớp sẽ vẽ heatmap trong quá trình quét
):
    if dt > dx**2 / (4 * alpha) or dt > dz**2 / (4 * alpha):
        print(f"Cảnh báo: dt = {dt} có thể quá lớn, nên < {min(dx**2 / (4 * alpha), dz**2 / (4 * alpha)):.2e}")

    Lx = nx * dx
    if t_scan_layer is None:
        t_scan_layer = Lx / laser_speed

    n_steps_scan = int(t_scan_layer / dt)
    n_steps_cool = int(t_cool_layer / dt)

    nz_current = nz_base
    T = np.full((nz_current, nx), T_ambient, dtype=float)
    figs = {}

    T_history_point_data = []
    time_data = []
    current_global_time = 0.0
    history_point_z = nz_base // 2
    history_point_x = nx // 2

    for layer_num in range(1, n_layers + 1):
        print(f"--- Simulating Layer {layer_num} ---")
        T_before_new_layer = T.copy()
        nz_new = nz_current + nz_layer
        T_new_array = np.full((nz_new, nx), T_ambient, dtype=float)
        if layer_num > 1:
            T_new_array[:nz_current, :] = T_before_new_layer
        else:
            T_new_array[:nz_current, :] = T
        T = T_new_array
        nz_current = nz_new

        # (A) Laser Scanning Phase
        print(f"Layer {layer_num}: Scanning...")
        for step in range(n_steps_scan):
            t_in_scan = step * dt
            laser_x_pos = (t_in_scan / t_scan_layer) * Lx
            T_next = T.copy()

            for i in range(1, nz_current - 1):
                for j in range(1, nx - 1):
                    T_next[i, j] = T[i, j] + alpha * dt * (
                        (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dx**2 +
                        (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dz**2
                    )
            T_next[0, :] = T_ambient
            T_next[:, 0] = T_ambient
            T_next[:, -1] = T_ambient
            # Bề mặt trên cùng sẽ được laser xử lý

            i_surf = nz_current - 1
            for j_col in range(nx):
                x_coord = j_col * dx
                if abs(x_coord - laser_x_pos) <= laser_radius:
                    T_next[i_surf, j_col] = target_laser_temp
                elif T_next[i_surf,j_col] < T_ambient : # Đảm bảo bề mặt không lạnh hơn môi trường
                     T_next[i_surf,j_col] = T_ambient


            T = T_next.copy()
            T_history_point_data.append(T[history_point_z, history_point_x])
            time_data.append(current_global_time + t_in_scan)

            # --- >>> VẼ HEATMAP TRONG KHI QUÉT <<< ---
            if layer_num == plot_during_scan_layer and step == n_steps_scan // 2: # Vẽ ở giữa quá trình quét lớp chỉ định
                fig_ds, ax_ds = plt.subplots(figsize=(8, 6))
                current_T_max_ds = T.max()
                plot_vmin_ds = T_ambient
                plot_vmax_ds = target_laser_temp + 100 # Ưu tiên hiển thị vùng 1000K
                if current_T_max_ds > plot_vmax_ds: # Nếu có điểm nào nóng hơn (do tích lũy)
                    plot_vmax_ds = current_T_max_ds + 50

                im_ds = ax_ds.imshow(T, cmap=colormap, origin='lower',
                                   extent=[0, nx * dx * 1000, 0, nz_current * dz * 1000],
                                   aspect='auto', vmin=plot_vmin_ds, vmax=plot_vmax_ds)
                ax_ds.set_title(f'Heatmap During Scan - Layer {layer_num}, Halfway (Laser at X={laser_x_pos*1000:.1f}mm)')
                ax_ds.set_xlabel('X (mm)')
                ax_ds.set_ylabel('Z (mm)')
                cbar_ds = fig_ds.colorbar(im_ds, ax=ax_ds)
                cbar_ds.set_label('Temperature (K)')
                plt.tight_layout()
                filename_ds = f'result_heatmap_layer{layer_num}_DURING_SCAN_target{int(target_laser_temp)}K.png'
                plt.savefig(filename_ds)
                figs[f'layer_{layer_num}_during_scan'] = fig_ds
                print(f"Saved heatmap during scan: {filename_ds}")
                plt.close(fig_ds)
        current_global_time += t_scan_layer

        # (B) Cooling Phase
        print(f"Layer {layer_num}: Cooling...")
        T_next_cool = T.copy()
        for step in range(n_steps_cool):
            t_in_cool = step*dt
            for i in range(1, nz_current - 1):
                for j in range(1, nx - 1):
                    T_next_cool[i, j] = T[i, j] + alpha * dt * (
                        (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / dx**2 +
                        (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / dz**2
                    )
            T_next_cool[0, :] = T_ambient
            T_next_cool[-1, :] = T_ambient
            T_next_cool[:, 0] = T_ambient
            T_next_cool[:, -1] = T_ambient
            T = T_next_cool.copy()
            T_history_point_data.append(T[history_point_z, history_point_x])
            time_data.append(current_global_time + t_in_cool)
        current_global_time += t_cool_layer

        if layer_num in plot_layers:
            fig, ax = plt.subplots(figsize=(8, 6))
            current_T_max = T.max()
            plot_vmin = T_ambient
            plot_vmax = target_laser_temp # Bắt đầu với nhiệt độ mục tiêu
            if current_T_max > plot_vmax: # Nếu nhiệt độ tối đa lớn hơn (do tích lũy)
                 plot_vmax = current_T_max + 50 # Thêm biên độ
            elif plot_vmax < T_ambient + 100: # Đảm bảo có một khoảng nhìn thấy
                 plot_vmax = T_ambient + 100
            if plot_vmax <= plot_vmin +10 : # Xử lý trường hợp không có sự thay đổi nhiều
                 plot_vmax = plot_vmin + 50


            im = ax.imshow(T, cmap=colormap, origin='lower',
                           extent=[0, nx * dx * 1000, 0, nz_current * dz * 1000],
                           aspect='auto', vmin=plot_vmin, vmax=plot_vmax)
            ax.set_title(f'Heatmap After Cooling - Layer {layer_num} (Max T: {current_T_max:.0f}K)')
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Z (mm)')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Temperature (K)')
            plt.tight_layout()
            filename = f'result_heatmap_layer_{layer_num}_AFTER_COOLING_target{int(target_laser_temp)}K.png'
            plt.savefig(filename)
            figs[f'layer_{layer_num}_after_cooling'] = fig
            print(f"Saved heatmap after cooling: {filename}")
            plt.close(fig)

    if T_history_point_data:
        fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
        ax_hist.plot(time_data, T_history_point_data)
        ax_hist.set_title(f'Temp History (X={history_point_x*dx*1000:.1f}mm, Z={history_point_z*dz*1000:.1f}mm from bottom)')
        ax_hist.set_xlabel('Global Time (s)')
        ax_hist.set_ylabel('Temperature (K)')
        ax_hist.grid(True)
        plt.tight_layout()
        filename_hist = f'result_temp_history_global_target{int(target_laser_temp)}K.png'
        plt.savefig(filename_hist)
        figs['history_global'] = fig_hist
        print(f"Saved global temperature history figure: {filename_hist}")
        plt.close(fig_hist)
    return figs

if __name__ == "__main__":
    generated_heatmaps = simulate_slm_layer_fdm_and_plot_heatmaps(
        nx=100, nz_layer=2, nz_base=10, n_layers=3,
        alpha=3e-6, # Giảm alpha hơn nữa
        dx=2.5e-5, dz=2.5e-5, # Lưới mịn hơn
        dt=1e-5,        # dt nhỏ hơn nữa cho ổn định và chi tiết
        laser_speed=0.05, # Laser di chuyển chậm hơn
        laser_radius=1e-4, # Bán kính laser nhỏ hơn (tập trung hơn)
        target_laser_temp=1000.0,
        t_cool_layer=0.02,
        plot_layers=[1, 2, 3],
        colormap='magma', # Thử magma
        plot_during_scan_layer=1 # Vẽ heatmap khi đang quét lớp 1
    )