import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class EggFreshnessModel(nn.Module):
    """ResNet 기반 계란 신선도 예측 모델 (검증용)"""
    def __init__(self, pretrained=False):
        super(EggFreshnessModel, self).__init__()
        
        from torchvision import models
        self.backbone = models.resnet50(pretrained=pretrained)
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

class TestDataset(Dataset):
    """테스트 데이터셋"""
    def __init__(self, image_paths, hu_values=None, transform=None):
        self.image_paths = image_paths
        self.hu_values = hu_values if hu_values is not None else [0] * len(image_paths)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        hu_value = self.hu_values[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (300, 400), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.FloatTensor([hu_value]), image_path

class EggFreshnessValidator:
    """계란 신선도 모델 검증 클래스"""
    
    def __init__(self, model_path, device=None):
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = self.load_model()
        
        # 전처리 변환
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Using device: {self.device}")
        print(f"Model loaded from: {model_path}")
    
    def load_model(self):
        """저장된 모델 로드"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        model = EggFreshnessModel(pretrained=False).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 모델 정보 출력
        print(f"Model validation loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"Model validation accuracy: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"Model timestamp: {checkpoint.get('timestamp', 'N/A')}")
        
        return model
    
    def validate_test_data(self, test_data_dir, excel_file_path=None):
        """저장된 테스트 데이터로 모델 검증"""
        print("\n" + "="*60)
        print("테스트 데이터 검증 시작")
        print("="*60)
        
        # 테스트 데이터 정보 로드
        test_info_path = os.path.join(test_data_dir, 'test_data_info.json')
        
        if not os.path.exists(test_info_path):
            print("Error: test_data_info.json not found!")
            return None
        
        with open(test_info_path, 'r', encoding='utf-8') as f:
            test_info = json.load(f)
        
        # 복사된 이미지 경로들
        test_image_paths = [info['copied_path'] for info in test_info]
        original_paths = [info['original_path'] for info in test_info]
        
        # 실제 H.U. 값들 추출 (엑셀 파일이 제공된 경우)
        true_hu_values = None
        if excel_file_path and os.path.exists(excel_file_path):
            true_hu_values = self.extract_hu_values_from_paths(original_paths, excel_file_path)
        
        # 데이터셋 및 데이터로더 생성
        test_dataset = TestDataset(test_image_paths, true_hu_values, self.transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        return self.evaluate_model(test_loader, true_hu_values is not None)
    
    def validate_custom_images(self, image_paths, true_hu_values=None):
        """사용자 지정 이미지들로 모델 검증"""
        print("\n" + "="*60)
        print("사용자 지정 이미지 검증 시작")
        print("="*60)
        
        # 이미지 경로들 확인
        valid_paths = []
        valid_hu_values = []
        
        for i, path in enumerate(image_paths):
            if os.path.exists(path):
                valid_paths.append(path)
                if true_hu_values:
                    valid_hu_values.append(true_hu_values[i])
            else:
                print(f"Warning: Image not found - {path}")
        
        if len(valid_paths) == 0:
            print("Error: No valid images found!")
            return None
        
        print(f"Valid images: {len(valid_paths)}")
        
        # 데이터셋 및 데이터로더 생성
        test_dataset = TestDataset(valid_paths, valid_hu_values if valid_hu_values else None, self.transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        return self.evaluate_model(test_loader, true_hu_values is not None)
    
    def evaluate_model(self, test_loader, has_true_values=True):
        """모델 평가 실행"""
        predictions = []
        true_values = []
        image_paths = []
        
        print("모델 예측 중...")
        
        with torch.no_grad():
            for batch_idx, (images, targets, paths) in enumerate(test_loader):
                images = images.to(self.device)
                
                # 예측 수행
                outputs = self.model(images)
                
                # 결과 저장
                predictions.extend(outputs.cpu().numpy().flatten())
                if has_true_values:
                    true_values.extend(targets.numpy().flatten())
                image_paths.extend(paths)
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed batch {batch_idx + 1}/{len(test_loader)}")
        
        predictions = np.array(predictions)
        true_values = np.array(true_values) if has_true_values else None
        
        print(f"총 {len(predictions)}개 이미지 예측 완료")
        
        # 결과 분석
        results = self.analyze_results(predictions, true_values, image_paths)
        
        # 결과 시각화
        if has_true_values:
            self.visualize_results(predictions, true_values)
        
        return results
    
    def analyze_results(self, predictions, true_values, image_paths):
        """결과 분석"""
        print("\n" + "-"*50)
        print("결과 분석")
        print("-"*50)
        
        results = {
            'predictions': predictions,
            'image_paths': image_paths,
            'prediction_stats': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'median': np.median(predictions)
            }
        }
        
        print(f"예측 H.U. 값 통계:")
        print(f"  평균: {results['prediction_stats']['mean']:.2f}")
        print(f"  표준편차: {results['prediction_stats']['std']:.2f}")
        print(f"  최솟값: {results['prediction_stats']['min']:.2f}")
        print(f"  최댓값: {results['prediction_stats']['max']:.2f}")
        print(f"  중앙값: {results['prediction_stats']['median']:.2f}")
        
        if true_values is not None:
            # 성능 메트릭 계산
            mse = mean_squared_error(true_values, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            
            results.update({
                'true_values': true_values,
                'metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2_score': r2
                }
            })
            
            print(f"\n성능 메트릭:")
            print(f"  MSE: {mse:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R² Score: {r2:.4f}")
            
            # 예측 정확도 범위별 분석
            error = np.abs(predictions - true_values)
            accuracy_ranges = [5, 10, 20, 50]
            
            print(f"\n예측 정확도 분석:")
            for range_val in accuracy_ranges:
                accuracy = np.mean(error <= range_val) * 100
                print(f"  ±{range_val} H.U. 내 정확도: {accuracy:.1f}%")
        
        return results
    
    def visualize_results(self, predictions, true_values):
        """결과 시각화"""
        print("\n결과 시각화 중...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 예측값 vs 실제값 산점도
        axes[0, 0].scatter(true_values, predictions, alpha=0.6)
        axes[0, 0].plot([true_values.min(), true_values.max()], 
                       [true_values.min(), true_values.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('실제 H.U. 값')
        axes[0, 0].set_ylabel('예측 H.U. 값')
        axes[0, 0].set_title('예측값 vs 실제값')
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² 값 표시
        r2 = r2_score(true_values, predictions)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12,
                       bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        # 2. 잔차 플롯
        residuals = predictions - true_values
        axes[0, 1].scatter(true_values, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('실제 H.U. 값')
        axes[0, 1].set_ylabel('잔차 (예측값 - 실제값)')
        axes[0, 1].set_title('잔차 플롯')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 오차 히스토그램
        error = np.abs(predictions - true_values)
        axes[1, 0].hist(error, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('절대 오차')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].set_title('절대 오차 분포')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 평균 절대 오차 표시
        mae = np.mean(error)
        axes[1, 0].axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE = {mae:.2f}')
        axes[1, 0].legend()
        
        # 4. 예측값과 실제값 분포 비교
        axes[1, 1].hist(true_values, bins=30, alpha=0.5, label='실제값', density=True)
        axes[1, 1].hist(predictions, bins=30, alpha=0.5, label='예측값', density=True)
        axes[1, 1].set_xlabel('H.U. 값')
        axes[1, 1].set_ylabel('밀도')
        axes[1, 1].set_title('실제값 vs 예측값 분포')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def extract_hu_values_from_paths(self, image_paths, excel_file_path):
        """이미지 경로에서 H.U. 값 추출"""
        print("이미지 경로에서 H.U. 값 추출 중...")
        
        # 엑셀 파일 로드
        excel_file = pd.ExcelFile(excel_file_path)
        all_data = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            df['date'] = sheet_name
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        hu_values = []
        for path in image_paths:
            # 경로에서 날짜, pan, num 추출
            path_parts = path.replace('\\', '/').split('/')
            
            try:
                date = None
                pan = None
                num = None
                
                for i, part in enumerate(path_parts):
                    if len(part) == 8 and part.isdigit():  # 날짜 형식
                        date = part
                    elif part.startswith('pan_'):
                        pan = int(part.split('_')[1])
                    elif len(part) == 2 and part.isdigit():  # num 폴더
                        num = int(part)
                        break
                
                if date and pan is not None and num is not None:
                    # 해당하는 H.U. 값 찾기
                    matching_row = combined_df[
                        (combined_df['date'] == date) & 
                        (combined_df['pan'] == pan) & 
                        (combined_df['num'] == num)
                    ]
                    
                    if not matching_row.empty:
                        hu_values.append(matching_row.iloc[0]['H.U.'])
                    else:
                        print(f"Warning: No matching H.U. value found for {path}")
                        hu_values.append(0)  # 기본값
                else:
                    print(f"Warning: Could not parse path {path}")
                    hu_values.append(0)  # 기본값
                    
            except Exception as e:
                print(f"Error processing path {path}: {e}")
                hu_values.append(0)  # 기본값
        
        return hu_values
    
    def predict_single_image(self, image_path):
        """단일 이미지 예측"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(image_tensor)
                
            return prediction.cpu().numpy()[0][0]
        
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None
    
    def generate_prediction_report(self, results, output_path=None):
        """예측 결과 리포트 생성"""
        print("\n" + "="*60)
        print("예측 결과 리포트 생성")
        print("="*60)
        
        # 데이터프레임 생성
        report_data = []
        for i, (pred, path) in enumerate(zip(results['predictions'], results['image_paths'])):
            row = {
                'Image_ID': i + 1,
                'Image_Path': path,
                'Predicted_HU': pred,
                'Image_Name': os.path.basename(path)
            }
            
            if 'true_values' in results:
                row['True_HU'] = results['true_values'][i]
                row['Absolute_Error'] = abs(pred - results['true_values'][i])
                row['Relative_Error_Percent'] = abs(pred - results['true_values'][i]) / results['true_values'][i] * 100
            
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        
        # 신선도 등급 추가 (H.U. 값에 따른)
        def classify_freshness(hu_value):
            if hu_value >= 72:
                return 'AA급 (매우 신선)'
            elif hu_value >= 60:
                return 'A급 (신선)'
            elif hu_value >= 45:
                return 'B급 (보통)'
            else:
                return 'C급 (신선도 낮음)'
        
        df_report['Freshness_Grade'] = df_report['Predicted_HU'].apply(classify_freshness)
        
        # 통계 요약
        print(f"총 검증 이미지 수: {len(df_report)}")
        print(f"신선도 등급별 분포:")
        grade_counts = df_report['Freshness_Grade'].value_counts()
        for grade, count in grade_counts.items():
            percentage = count / len(df_report) * 100
            print(f"  {grade}: {count}개 ({percentage:.1f}%)")
        
        # 파일 저장
        if output_path:
            df_report.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"\n리포트 저장됨: {output_path}")
        
        return df_report

# 검증 실행 함수들
def validate_saved_test_data(model_path, test_data_dir, excel_file_path=None):
    """저장된 테스트 데이터로 검증"""
    validator = EggFreshnessValidator(model_path)
    results = validator.validate_test_data(test_data_dir, excel_file_path)
    
    if results:
        # 리포트 생성
        report_path = os.path.join(os.path.dirname(test_data_dir), 'validation_report.csv')
        df_report = validator.generate_prediction_report(results, report_path)
        
        return results, df_report
    
    return None, None

def validate_custom_images(model_path, image_paths, true_hu_values=None):
    """사용자 지정 이미지들로 검증"""
    validator = EggFreshnessValidator(model_path)
    results = validator.validate_custom_images(image_paths, true_hu_values)
    
    if results:
        # 리포트 생성
        output_dir = os.path.dirname(model_path)
        report_path = os.path.join(output_dir, 'custom_validation_report.csv')
        df_report = validator.generate_prediction_report(results, report_path)
        
        return results, df_report
    
    return None, None

def predict_single_image(model_path, image_path):
    """단일 이미지 예측"""
    validator = EggFreshnessValidator(model_path)
    prediction = validator.predict_single_image(image_path)
    
    if prediction is not None:
        # 신선도 등급 판정
        if prediction >= 72:
            grade = 'AA급 (매우 신선)'
        elif prediction >= 60:
            grade = 'A급 (신선)'
        elif prediction >= 45:
            grade = 'B급 (보통)'
        else:
            grade = 'C급 (신선도 낮음)'
        
        print(f"\n예측 결과:")
        print(f"이미지: {os.path.basename(image_path)}")
        print(f"예측 H.U. 값: {prediction:.2f}")
        print(f"신선도 등급: {grade}")
        
        return prediction, grade
    
    return None, None

# 사용 예시
if __name__ == "__main__":
    # 예시 1: 저장된 테스트 데이터로 검증
    MODEL_PATH = "./trained_models/egg_freshness_model_epoch50_val_acc0.8500_val_loss0.1200_20250810_143000.pth"
    TEST_DATA_DIR = "./trained_models/test_data_20250810_143000"
    EXCEL_FILE_PATH = "path/to/your/excel_file.xlsx"
    
    print("=== 저장된 테스트 데이터 검증 ===")
    results, report_df = validate_saved_test_data(MODEL_PATH, TEST_DATA_DIR, EXCEL_FILE_PATH)
    
    # 예시 2: 사용자 지정 이미지들로 검증
    custom_images = [
        "path/to/custom/image1.bmp",
        "path/to/custom/image2.bmp",
        "path/to/custom/image3.bmp"
    ]
    custom_hu_values = [65.5, 72.3, 45.8]  # 실제 H.U. 값들 (있는 경우)
    
    print("\n=== 사용자 지정 이미지 검증 ===")
    custom_results, custom_report = validate_custom_images(MODEL_PATH, custom_images, custom_hu_values)
    
    # 예시 3: 단일 이미지 예측
    single_image_path = "path/to/single/image.bmp"
    
    print("\n=== 단일 이미지 예측 ===")
    prediction, grade = predict_single_image(MODEL_PATH, single_image_path)