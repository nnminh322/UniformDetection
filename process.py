import numpy as np

class Process:
    def __init__(self, res1, res2, frame_height=None, w_model=0.6, w_logic=0.4, final_thresh=0.5, roi_thresh=0.3):
        self.frame_height = frame_height
        self.roi_thresh = roi_thresh
        self.roi_y_min = frame_height * roi_thresh if frame_height else 0
        
        self.conf1 = res1.boxes.conf.cpu().numpy() if len(res1.boxes) > 0 else np.array([])
        self.cls1 = res1.boxes.cls.cpu().numpy() if len(res1.boxes) > 0 else np.array([])
        self.xyxy1 = res1.boxes.xyxy.cpu().numpy() if len(res1.boxes) > 0 else np.array([])
        
        self.cls2 = res2.boxes.cls.cpu().numpy() if len(res2.boxes) > 0 else np.array([])
        self.xyxy2 = res2.boxes.xyxy.cpu().numpy() if len(res2.boxes) > 0 else np.array([])
        
        self._apply_roi_filter()
        
        self.w_model = w_model
        self.w_logic = w_logic
        self.final_thresh = final_thresh
        
        self.final_hat_detections = []  
    
    def _apply_roi_filter(self):
        if self.frame_height is None:
            return
        
        if len(self.xyxy1) > 0:
            mask1 = []
            for box in self.xyxy1:
                y_center = (box[1] + box[3]) / 2
                mask1.append(y_center > self.roi_y_min)
            mask1 = np.array(mask1)
            self.xyxy1 = self.xyxy1[mask1]
            self.conf1 = self.conf1[mask1]
            self.cls1 = self.cls1[mask1]
        
        if len(self.xyxy2) > 0:
            mask2 = []
            for box in self.xyxy2:
                y_center = (box[1] + box[3]) / 2
                mask2.append(y_center > self.roi_y_min)
            mask2 = np.array(mask2)
            self.xyxy2 = self.xyxy2[mask2]
            self.cls2 = self.cls2[mask2]
        
    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0
    
    def check_spatial_logic(self, hat_box, uniform_boxes):
        hx1, hy1, hx2, hy2 = hat_box
        h_center_x = (hx1 + hx2) / 2
        h_height = hy2 - hy1

        best_logic_score = 0.1  
        
        if len(uniform_boxes) == 0:
            return best_logic_score

        for u_box in uniform_boxes:
            ux1, uy1, ux2, uy2 = u_box
            u_center_x = (ux1 + ux2) / 2
            u_width = ux2 - ux1
            
            delta_x = abs(h_center_x - u_center_x)
            is_aligned_x = delta_x < (u_width * 0.5)

            if is_aligned_x:
                return 1.0 

        return best_logic_score
    
    def apply_ensemble_logic(self):

        self.final_hat_detections = []
        
        if len(self.xyxy1) == 0:
            return
        
        for box, conf, cls in zip(self.xyxy1, self.conf1, self.cls1):
            if cls != 0:
                return 
            logic_score = self.check_spatial_logic(box, self.xyxy2)
            
            final_score = (conf * self.w_model) + (logic_score * self.w_logic)
 
            if final_score >= self.final_thresh:
                self.final_hat_detections.append((box, cls, final_score, logic_score))
    
    def get_final_detections(self):
        uniform_detections = [(box, cls) for box, cls in zip(self.xyxy2, self.cls2)]
        return self.final_hat_detections, uniform_detections
