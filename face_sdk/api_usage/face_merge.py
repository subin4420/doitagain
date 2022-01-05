from PIL import Image
class merging_class:
    def merge(self):

        image1 = Image.open('C:/Users/kii/Desktop/FaceX-Zoo-main/face_sdk/api_usage/test_images/pic1.jpg')
        image2 = Image.open('C:/Users/kii/Desktop/FaceX-Zoo-main/face_sdk/api_usage/test_images/pic2.jpg')
        image1 = image1.resize((224, 224))
        image2 = image2.resize((224, 224))
        image1_size = image1.size
        image2_size = image2.size
        new_image = Image.new('RGB', (2 * image1_size[0], image1_size[1]), (500, 500, 500))
        new_image.paste(image1, (0, 0))
        new_image.paste(image2, (image1_size[0], 0))
        new_image.save("C:/Users/kii/Desktop/FaceX-Zoo-main/face_sdk/api_usage/hallway/merged.jpg")
        print("merge finished")


