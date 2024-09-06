import pandas as pd
import os

# 엑셀 파일 경로 목록
excel_file_paths = [f'/Users/moonsun/LLMBootcamp/숙박추천/data/xls/{i}.xls' for i in range(1, 16)]
# 변환된 CSV 파일이 저장될 폴더 경로
csv_folder_path = '/Users/moonsun/LLMBootcamp/숙박추천/data/csv'
# 최종 합쳐진 CSV 파일 경로
combined_csv_file_path = '/Users/moonsun/LLMBootcamp/숙박추천/data/combined.csv'

# 제거할 열 번호 (0-based index)
columns_to_remove = [1,2,5,6,8,9] + list(range(22, 33))

# 빈 데이터프레임 생성
combined_df = pd.DataFrame()

# 엑셀 파일을 읽고 지정된 열을 제거한 후 CSV 파일로 저장
for excel_file_path in excel_file_paths:
    # .xls 파일을 읽기
    df = pd.read_excel(excel_file_path, engine='xlrd')
    
    # 0-based index를 사용하여 열 이름으로 변환
    columns_to_remove_names = [df.columns[i] for i in columns_to_remove if i < len(df.columns)]
    
    # 지정된 열 제거
    df = df.drop(columns=columns_to_remove_names)
    
    # 모든 데이터를 문자열로 변환한 후 ','를 '/'로 변환
    df = df.astype(str).applymap(lambda x: x.replace(',', '/'))
    
    # CSV 파일 경로 설정
    csv_file_path = os.path.join(csv_folder_path, os.path.basename(excel_file_path).replace('.xls', '.csv'))
    
    # 필터링된 데이터프레임을 CSV 파일로 저장 (utf-8-sig 인코딩 사용)
    df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
    
    # 합쳐진 데이터프레임에 현재 데이터프레임 추가
    combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    # 저장된 파일 확인 출력
    print(f"Saved filtered CSV file to {csv_file_path}")

# Q부터 AD까지의 열 제거 (0-based index: 16부터 29)
columns_to_remove_combined = list(range(16, 30))
columns_to_remove_combined_names = [combined_df.columns[i] for i in columns_to_remove_combined]
combined_df = combined_df.drop(columns=columns_to_remove_combined_names)

# 최종 합쳐진 CSV 파일 저장
combined_df.to_csv(combined_csv_file_path, index=False, encoding='utf-8-sig')

# 합쳐진 데이터프레임 출력
print(combined_df.head())
