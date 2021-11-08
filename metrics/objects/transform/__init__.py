from .transformers import (Box2DToKeyPointsWithCenter, Box2DToKeyPoints,
                           LshapeToKeyPoints, MinimalBox2D)

TRANSORM_FACTORY = {
    'Box2DToKeyPoints': Box2DToKeyPoints(),
    'LshapeToKeyPoints': LshapeToKeyPoints(),
    'MinimalBox2D': MinimalBox2D(),
    'Box2DToKeyPointsWithCenter': Box2DToKeyPointsWithCenter()
}
