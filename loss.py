import tensorflow as tf

def yolo_loss (y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1e6)
    obj_mask = y_true[..., 4:5]
    
    true_xy = y_true[..., 0:2]
    pred_xy = y_pred[..., 0:2]

    true_wh = y_true[..., 2:4]
    pred_wh = y_pred[..., 2:4]

    xy_loss = tf.square(true_xy - pred_xy)
    wh_loss = tf.square(tf.sqrt(true_wh + 1e-6) - tf.sqrt(pred_wh + 1e-6))

    true_conf = y_true[..., 4]
    pred_conf = y_pred[...,4]
    obj_mask = y_true[...,4:5]
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_coord = 5.0

    coord_loss = tf.reduce_sum(obj_mask*(xy_loss + wh_loss))
    coord_loss = lambda_coord * coord_loss

    conf_loss = (lambda_obj * obj_mask *tf.square(true_conf - pred_conf) + 
                 lambda_noobj * (1-obj_mask) * tf.square(true_conf - pred_conf))
    conf_loss = tf.reduce_sum(conf_loss)

    class_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 10:] - y_pred[..., 10:]))

    loss = coord_loss + conf_loss + class_loss
    loss = loss / tf.cast(tf.shape(y_true)[0] , tf.float32)
    return loss


