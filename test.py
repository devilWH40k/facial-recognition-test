from deepface import DeepFace


def main(file):
    try:
        embedding_objs = DeepFace.represent(img_path = file, model_name="SFace")
        print(len(embedding_objs[0]["embedding"]))
        print("Cooool!!!")
    except Exception as e:
        print("[still corrupted]")
        print(e)

main("./dataset\zhdanov\zhdanov_10.jpg")