from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"E:\PythonLab\Huaqing_Ruiyang\pythonProject\Project2_Day5\ultralytics-main\runs\detect\train2\weights\best.pt")
    model.val(data="coco8.yaml",imgsz=640,epochs=20,batch=16)