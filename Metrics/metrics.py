from torchmetrics import Dice

def calculate_dice_per_class(pred, target, num_classes):
    dice_scores = []

    for cls in range(num_classes):
        # Créez des masques binaires pour la classe actuelle
        pred_cls = (pred == cls).long()
        target_cls = (target == cls).long()
        
        dice_metric = Dice().to(pred.device)
        
        dice_score = dice_metric(pred_cls, target_cls)
        dice_scores.append((cls, dice_score.item()))

    return dice_scores