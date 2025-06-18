# run_sr.jl (Phiên bản sửa lỗi MethodError và parallelism)

using SymbolicRegression
using DataFrames
using CSV
using Random

println("Bắt đầu script Symbolic Regression bằng Julia...")

# !!! QUAN TRỌNG: THAY ĐỔI ĐƯỜNG DẪN NÀY CHO PHÙ HỢP !!!
csv_filepath = joinpath(@__DIR__, "output_SR_results_julia", "C:\\Users\\MAY02\\Documents\\Physics_Symbolic_regression\\output_SR_results\\relax_Al_8line_1_processed.csv")
# csv_filepath = "C:/Users/MAY02/Documents/Physics_Symbolic_regression/output_SR_results_julia/data_for_julia_sr.csv" # VÍ DỤ

target_column_name = "Temperature"

println("Sẽ tải dữ liệu từ: ", csv_filepath)

try
    if !isfile(csv_filepath)
        println("LỖI: Không tìm thấy file dữ liệu CSV tại: '", csv_filepath, "'")
        exit()
    end

    df = CSV.read(csv_filepath, DataFrame)
    println("Tải dữ liệu thành công. Số dòng ban đầu: ", nrow(df), ", Số cột: ", ncol(df))

    available_columns = names(df)
    println("Các cột có sẵn trong DataFrame vừa đọc: ", available_columns)

    # println("\nKiểu dữ liệu ban đầu của các cột:")
    # for col_name in available_columns
    #     println("Cột '", col_name, "': ", eltype(df[!, Symbol(col_name)]))
    # end

    if nrow(df) == 0; println("LỖI: DataFrame rỗng sau khi tải."); exit(); end
    if !(target_column_name in available_columns); println("LỖI: Cột target '", target_column_name, "' không tìm thấy."); exit(); end

    # --- XỬ LÝ MISSING VÀ CHUYỂN ĐỔI KIỂU DỮ LIỆU ---
    feature_column_names_str = [col for col in available_columns if col != target_column_name]
    if isempty(feature_column_names_str); println("LỖI: Không có cột feature nào."); exit(); end
    
    all_used_column_names_str = vcat(feature_column_names_str, target_column_name)
    all_used_column_symbols = Symbol.(all_used_column_names_str)
    # println("\nCác cột sẽ được sử dụng cho SR: ", all_used_column_names_str)

    df_cleaned = dropmissing(df, all_used_column_symbols)
    rows_dropped = nrow(df) - nrow(df_cleaned)
    if rows_dropped > 0; println("Đã loại bỏ ", rows_dropped, " hàng do có giá trị missing."); end
    if nrow(df_cleaned) == 0; println("LỖI: DataFrame rỗng sau khi loại bỏ missing."); exit(); end
    println("Số dòng sau khi làm sạch missing: ", nrow(df_cleaned))

    local y_vec::Vector{Float64} # Đổi tên để rõ ràng hơn và đảm bảo là Vector
    local X_mat::Matrix{Float64} # Đổi tên để rõ ràng hơn

    try
        y_temp_col = df_cleaned[!, Symbol(target_column_name)]
        y_vec = Vector{Float64}(y_temp_col)
    catch e
        println("LỖI khi chuyển đổi cột target '$target_column_name' sang Vector{Float64}."); showerror(stdout, e); println(); exit()
    end

    feature_column_symbols_cleaned = Symbol.(feature_column_names_str)
    try
        num_features = length(feature_column_symbols_cleaned)
        num_samples_cleaned = nrow(df_cleaned)
        X_matrix_temp = Matrix{Float64}(undef, num_features, num_samples_cleaned)
        for (i, feature_sym) in enumerate(feature_column_symbols_cleaned)
            X_matrix_temp[i, :] = Vector{Float64}(df_cleaned[!, feature_sym])
        end
        X_mat = X_matrix_temp
    catch e
        println("LỖI khi chuyển đổi các cột feature sang Matrix{Float64}."); showerror(stdout, e); println(); exit()
    end
    
    # println("\nĐã chuyển đổi X và y sang kiểu dữ liệu cụ thể (Float64).")
    # println("Kiểu dữ liệu của y_vec: ", eltype(y_vec), ", kích thước: ", size(y_vec))
    # println("Kiểu dữ liệu của X_mat: ", eltype(X_mat), ", kích thước (features, samples): ", size(X_mat))

    if size(X_mat,2) != length(y_vec) || length(y_vec) == 0; println("LỖI: Dữ liệu X_mat hoặc y_vec không hợp lệ sau khi xử lý."); exit(); end

    # --- 2. Thiết lập Tùy chọn cho Symbolic Regression ---
    println("\nThiết lập các tùy chọn cho Symbolic Regression...")
    custom_square(x) = x^2
    custom_cube(x) = x^3
    options = Options(
        binary_operators=[+, -, *, /, ^],
        unary_operators=[cos, sin, exp, log, sqrt, abs, tanh, inv, custom_square, custom_cube],
        verbosity=1, # Hoặc 0 để ít log hơn, 2 để nhiều hơn
        progress=true,
        populations=100,
        parsimony=0.0032, # Thử nghiệm với giá trị này
    )

    var_names_for_search = feature_column_names_str
    println("Sử dụng tên biến cho EquationSearch: ", var_names_for_search)

    # --- 3. Chạy Symbolic Regression (ĐÃ SỬA) ---
    println("\nBắt đầu quá trình tìm kiếm phương trình (EquationSearch)...")
    niterations = 20 

    Random.seed!(42)

    # --- DEBUG: Kiểm tra kiểu và kích thước của X_mat và y_vec ngay trước khi gọi EquationSearch ---
    println("Trước EquationSearch - Kiểu X_mat: ", typeof(X_mat), ", Kích thước X_mat: ", size(X_mat))
    println("Trước EquationSearch - Kiểu y_vec: ", typeof(y_vec), ", Kích thước y_vec: ", size(y_vec))
    # --- KẾT THÚC DEBUG ---

    hall_of_fame = EquationSearch(
        X_mat, y_vec, # Sử dụng X_mat và y_vec đã được đảm bảo kiểu
        niterations=niterations,
        options=options,
        variable_names=var_names_for_search,
        parallelism=:serial # Sử dụng :serial để chạy tuần tự và tránh lỗi luồng.
                            # Nếu muốn đa luồng, chạy julia --threads auto symbolic.jl và đặt parallelism=:multithreading
    )
    println("\n✅ Hoàn thành EquationSearch.")

    # --- 4. Hiển thị và Lưu Kết quả ---
    # (Giữ nguyên phần này như code đầy đủ trước đó, đảm bảo sử dụng hall_of_fame)
    println("\nCác phương trình trong Hall of Fame (Pareto frontier):")
    pareto_frontier_df = dominating_pareto_frontier(hall_of_fame)

    if nrow(pareto_frontier_df) > 0
        for i in 1:nrow(pareto_frontier_df)
            eq_row = pareto_frontier_df[i, :]
            score_str = "score" in names(eq_row) ? "| Score: $(round(eq_row.score, digits=5))" : ""
            println("Equation ", i, ": ", eq_row.equation,
                    " | Loss: ", round(eq_row.loss, digits=5),
                    " | Complexity: ", eq_row.complexity,
                    " $score_str")
        end

        best_overall_eq_row = pareto_frontier_df[1, :] 
        if "score" in names(pareto_frontier_df) && nrow(pareto_frontier_df) > 0
            sorted_by_score_df = sort(pareto_frontier_df, :score, rev=true)
            if nrow(sorted_by_score_df) > 0; best_overall_eq_row = sorted_by_score_df[1, :]; end
        elseif nrow(pareto_frontier_df) > 0 
             best_overall_eq_row = pareto_frontier_df[argmin(pareto_frontier_df.loss), :]
        end

        println("\n--- Phương trình tốt nhất được chọn ---")
        println("Equation: ", best_overall_eq_row.equation)
        println("Loss: ", best_overall_eq_row.loss)
        println("Complexity: ", best_overall_eq_row.complexity)
        if "score" in names(best_overall_eq_row); println("Score: ", best_overall_eq_row.score); end

        output_dir_julia = dirname(csv_filepath) 
        output_file_julia_results = joinpath(output_dir_julia, "best_symbolic_expression_julia_full_v4.txt") # Đổi tên file output
        try
            open(output_file_julia_results, "w") do f
                write(f, "Các phương trình trong Pareto Frontier (SymbolicRegression.jl):\n")
                for i in 1:nrow(pareto_frontier_df)
                    eq_row = pareto_frontier_df[i, :]
                    score_str = "score" in names(eq_row) ? "| Score: $(round(eq_row.score, digits=5))" : ""
                    write(f, "Eq $i: $(eq_row.equation) | Loss: $(round(eq_row.loss, digits=5)) | Complexity: $(eq_row.complexity) $score_str\n")
                end
                write(f, "\n--- Phương trình tốt nhất được chọn ---\n")
                write(f, "Equation: $(best_overall_eq_row.equation)\n")
                # ... (ghi các thông tin khác như trước)
            end
            println("\n📄 Kết quả đã được lưu vào: ", output_file_julia_results)
        catch e_file; println("LỖI khi ghi file kết quả: ", e_file); end
    else
        println("\n⚠️ Không tìm thấy phương trình nào trong Hall of Fame.")
    end

catch e
    println("\nĐÃ XẢY RA LỖI TRONG QUÁ TRÌNH CHẠY SCRIPT JULIA:")
    showerror(stdout, e)
    println()
    # current_exceptions_stack = Base.catch_stack()
    # if !isempty(current_exceptions_stack); println("\n--- Backtrace ---"); for (exc, bt) in current_exceptions_stack; showerror(stdout, exc, bt); println(stdout); end; end
end

println("\n🏁 Script Julia hoàn thành.")