from facenet_pytorch import MTCNN
import os
from PIL import Image



mtcnn = MTCNN(image_size=224, margin=20, keep_all=False)

def crop_and_save(src_root, dst_root):
    for split in ["Train", "Validation"]:
        for cls in ["deepfake", "original"]:
            src_dir = os.path.join(src_root, split, cls)
            dst_dir = os.path.join(dst_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            
            saved, skipped = 0, 0
            for fname in os.listdir(src_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    continue
                img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
                face = mtcnn(img)  # returns tensor or None
                if face is not None:
                    # convert back to PIL and save
                    face_pil = Image.fromarray(
                        (face.permute(1,2,0).numpy() * 128 + 127.5).clip(0,255).astype("uint8")
                    )
                    face_pil.save(os.path.join(dst_dir, fname))
                    saved += 1
                else:
                    skipped += 1
            
            print(f"{split}/{cls}: saved {saved}, no face detected {skipped}")

crop_and_save(
    "/home/admin1/Delete_when_done/data/dataset",
    "/home/admin1/Delete_when_done/croped_data"
)