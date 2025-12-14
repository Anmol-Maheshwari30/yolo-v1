import tensorflow as tf
"""
def yolo_loss(y_true, y_pred):
    coord_loss = tf.reduce_sum(tf.square(y_true[..., :4] - y_pred[..., :4]))
    obj_loss = tf.reduce_sum(tf.square(y_true[...,4] - y_pred[..., 4]))
    class_loss = tf.reduce_sum(tf.square(y_true[...,10:] - y_pred[...,10:]))
    return coord_loss+obj_loss+class_loss
"""
def yolo_loss(y_true, y_pred):
    obj_mask = y_true[..., 4:5]
    coord_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 0:4] - y_pred[..., 0:4]))
    # Objectness loss
    obj_loss = tf.reduce_sum(tf.square(y_true[..., 4] - y_pred[..., 4]))
    # Classification loss
    class_loss = tf.reduce_sum(obj_mask * tf.square(y_true[..., 10:] - y_pred[..., 10:]))
    return coord_loss + obj_loss + class_loss
