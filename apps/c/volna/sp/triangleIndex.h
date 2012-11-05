inline void triangleIndex(float *val, float* x, float* y, float* nodeCoordsA, float* nodeCoordsB, float* nodeCoordsC, float* values) {
  // Return value on cell if the given point is inside the cell
  bool isInside = false;

  // First, check if the point is in the bounding box of the triangle
  // vertices (else, the algorithm is not nearly robust enough)
  float xmin = MIN(MIN(nodeCoordsA[0], nodeCoordsB[0]), nodeCoordsC[0]);
  float xmax = MAX(MAX(nodeCoordsA[0], nodeCoordsB[0]), nodeCoordsC[0]);
  float ymin = MIN(MIN(nodeCoordsA[1], nodeCoordsB[1]), nodeCoordsC[1]);
  float ymax = MAX(MAX(nodeCoordsA[1], nodeCoordsB[1]), nodeCoordsC[1]);

  if ( ( *x < xmin ) || ( *x > xmax ) ||
      ( *y < ymin ) || ( *y > ymax ) ) {
    isInside = false;
  }else{
    // Case where the point is in the bounding box. Here, if abc is not
    // Check if the Triangle vertices are clockwise or
    // counter-clockwise
    float insider = 1.0f;
    float p[2] = {*x, *y};

#define ORIENT2D(pA, pB, pC) (pA[0] - pC[0]) * (pB[1] - pC[1]) - (pA[1] - pC[1]) * (pB[0] - pC[0])
    if ( ORIENT2D(nodeCoordsA, nodeCoordsB, nodeCoordsC) > 0 ) {  // counter clockwise
      insider =  ORIENT2D( nodeCoordsA, p, nodeCoordsC);
      insider *= ORIENT2D( nodeCoordsA, nodeCoordsB, p);
      insider *= ORIENT2D( nodeCoordsB, nodeCoordsC, p);
    }
    else {      // clockwise
      insider =  ORIENT2D( nodeCoordsA, p, nodeCoordsB);
      insider *= ORIENT2D( nodeCoordsA, nodeCoordsC, p);
      insider *= ORIENT2D( nodeCoordsC, nodeCoordsB, p);
    }
    isInside = insider > 0.0f;
  }

  if ( isInside )
    *val = values[0] + values[3]; // H + Zb
}

//float orient2d(float* pA, float* pB, float* pC) {
//  return (pA[0] - pC[0]) * (pB[1] - pC[1]) - (pA[1] - pC[1]) * (pB[0] - pC[0]);
//}
//
//// Is the point p inside the triangle abc ? the z component is discarded
//bool isInside( float* x, float* y, float* nodeCoordsA, float* nodeCoordsB, float* nodeCoordsC) {
//  // First, check if the point is in the bounding box of the triangle
//  // vertices (else, the algorithm is not nearly robust enough)
//  float xmin = MIN(MIN(nodeCoordsA[0], nodeCoordsB[0]), nodeCoordsC[0]);
//  float xmax = MAX(MAX(nodeCoordsA[0], nodeCoordsB[0]), nodeCoordsC[0]);
//  float ymin = MIN(MIN(nodeCoordsA[1], nodeCoordsB[1]), nodeCoordsC[1]);
//  float ymax = MAX(MAX(nodeCoordsA[1], nodeCoordsB[1]), nodeCoordsC[1]);
//
//  if ( ( *x < xmin ) || ( *x > xmax ) ||
//       ( *y < ymin ) || ( *y > ymax ) )
//    return false;
//
//  // Case where the point is in the bounding box. Here, if abc is not
//  //
//  // Check if the Triangle vertices are clockwise or
//  // counter-clockwise
//  float insider = 1.0f;
//  float p[2] = {*x, *y};
//  if ( orient2d(nodeCoordsA, nodeCoordsB, nodeCoordsC) > 0 ) {  // counter clockwise
//    insider =  orient2d( nodeCoordsA, p, nodeCoordsC);
//    insider *= orient2d( nodeCoordsA, nodeCoordsB, p);
//    insider *= orient2d( nodeCoordsB, nodeCoordsC, p);
//  }
//
//  else {      // clockwise
//    insider =  orient2d( nodeCoordsA, p, nodeCoordsB);
//    insider *= orient2d( nodeCoordsA, nodeCoordsC, p);
//    insider *= orient2d( nodeCoordsC, nodeCoordsB, p);
//  }
//
//  return (insider > 0.);
//}
//
//// Return value on cell if the given point is inside the cell
//inline void triangleIndex(float *val, float* x, float* y, float* nodeCoordsA, float* nodeCoordsB, float* nodeCoordsC, float* values) {
//  if ( isInside(x, y, nodeCoordsA, nodeCoordsB, nodeCoordsC) )
//    *val = values[0] + values[3]; // H + Zb
//}
