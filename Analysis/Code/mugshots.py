import os
import cv2

def extract_mugshots(root_dir, output_dir="mugshots"):
    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Traverse each subdirectory in root_dir
    for sub_name in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub_name)
        if not os.path.isdir(sub_path):
            continue  # skip files

        # Find mp4 file with 'How-are-you' in its name
        for file in os.listdir(sub_path):
            if file.endswith('.mp4') and 'How-are-you' in file:
                video_path = os.path.join(sub_path, file)

                # Load video
                cap = cv2.VideoCapture(video_path)
                success, frame = cap.read()
                cap.release()

                if success:
                    # Save first frame as JPEG
                    out_file = os.path.join(output_dir, f"{sub_name}.jpg")
                    cv2.imwrite(out_file, frame)
                    print(f"Saved: {out_file}")
                else:
                    print(f"Failed to read frame from {video_path}")
                break  # Assuming only one matching video per subdirectory

if __name__ == "__main__":

    in_folder = f"Analysis/Raw/Trash/"
    out_folder = f"Analysis/Results/Trash/"

    extract_mugshots(in_folder, output_dir=out_folder)