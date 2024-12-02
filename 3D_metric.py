import os
import numpy as np
from tqdm import tqdm

# Định nghĩa hàm SSIM
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    from scipy.ndimage import gaussian_filter

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    
    mu1_sq = np.multiply(mu1, mu1)
    mu2_sq = np.multiply(mu2, mu2)
    mu1_mu2 = np.multiply(mu1, mu2)

    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_num / ssim_den
    
    return np.mean(ssim_map)

# Định nghĩa hàm MAE
def compute_mae(image1, image2):
    return np.mean(np.abs(image1 - image2))

# Định nghĩa hàm PSNR
def compute_psnr(image1, image2, data_range=1.0):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Trường hợp ảnh giống nhau hoàn toàn
    psnr_value = 10 * np.log10((data_range ** 2) / mse)
    return psnr_value

# Đường dẫn tới các thư mục
gt_folder = "/workdir/radish/PET-CT/data_gen/3D_LABELED_CT_PET_V2"
# pre_folder = "/home/PET-CT/thaind/3D_BBDM/BBDM_folk/results/3D_BBDM_s=4/LBBDM-f4/sample_to_eval/200"
# pre_folder = "/home/PET-CT/CPDM_samples"
# pre_folder = "/workdir/carrot/PET-CT-Research/3D_visualization/3D_PET"
pre_folder = "/workdir/radish/PET-CT/3D_reggan/metric"

# Danh sách để lưu trữ các giá trị metrics
ssim_scores = []
mae_scores = []
psnr_scores = []

# Duyệt qua từng thư mục con trong thư mục ground truth
for subfolder in tqdm(os.listdir(gt_folder)):
    gt_subfolder_path = os.path.join(gt_folder, subfolder)
    pre_subfolder_path = os.path.join(pre_folder, subfolder)

    # Kiểm tra xem cả hai thư mục con có tồn tại không
    if os.path.isdir(gt_subfolder_path) and os.path.isdir(pre_subfolder_path):
        # Đọc các tệp trong mỗi thư mục con
        gt_file_path = os.path.join(gt_subfolder_path, "pet.npy")
        pre_file_path = os.path.join(pre_subfolder_path, "predicted_volume.npy")
        
        # Đảm bảo rằng các tệp dự kiến tồn tại
        if os.path.exists(gt_file_path) and os.path.exists(pre_file_path):
            try:
                # Tải các hình ảnh dưới dạng mảng numpy
                gt_img = np.load(gt_file_path, allow_pickle=True)
                pre_img = np.load(pre_file_path, allow_pickle=True)
            except:
                continue

            # Chuẩn hóa các hình ảnh
            pre_img = pre_img / 32767.0
            gt_img = gt_img / 32767.0
            
            # pre_img = pre_img[:, :, :, 0]  # Chỉ sử dụng kênh đầu tiên
            pre_img = pre_img.astype(np.float32)
            gt_img = gt_img.astype(np.float32)
            
            # Đảm bảo pre_img và gt_img có cùng kích thước và đúng số chiều
            if pre_img.shape != gt_img.shape or len(pre_img.shape) != 3:
                print(pre_img.shape, gt_img.shape)
                continue

            # Tính toán SSIM cho toàn bộ khối 3D
            ssim_value = ssim_exact(gt_img, pre_img)

            # Tính toán MAE cho toàn bộ khối 3D
            mae_value = compute_mae(gt_img, pre_img)

            # Tính toán PSNR cho toàn bộ khối 3D
            psnr_value = compute_psnr(gt_img, pre_img, data_range=1.0)

            # Lưu kết quả
            ssim_scores.append(ssim_value)
            mae_scores.append(mae_value)
            psnr_scores.append(psnr_value)

            # print(f"Shapes: {pre_img.shape} | SSIM: {ssim_value:.4f}, MAE: {mae_value:.4f}, PSNR: {psnr_value:.4f}")
            
            # break  # Xóa dòng này nếu bạn muốn xử lý tất cả các thư mục con

# Tính điểm trung bình của các metric trên tất cả các cặp
overall_mean_ssim = np.mean(ssim_scores)
overall_mean_mae = np.mean(mae_scores)
overall_mean_psnr = np.mean(psnr_scores)

# In kết quả trung bình cho tất cả các khối đã đánh giá
print("Overall Mean SSIM: {:.4f}".format(overall_mean_ssim))
print("Overall Mean MAE: {:.4f}".format(overall_mean_mae * 32767))
print("Overall Mean PSNR: {:.4f}".format(overall_mean_psnr))


# 3D_model_overlap_3 
# Overall Mean SSIM: 0.9326
# Overall Mean MAE: 303.5296
# Overall Mean PSNR: 29.3910

# 3D_model_overlap_5
# Overall Mean SSIM: 0.9330
# Overall Mean MAE: 302.4300
# Overall Mean PSNR: 29.4173

# non_overlap 
# Overall Mean SSIM: 0.9317
# Overall Mean MAE: 306.0933
# Overall Mean PSNR: 29.3246