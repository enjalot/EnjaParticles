


//column major 4x4 * 4x1
float4 matmult4(constant float* mat, float4 vec)
{
    float4 rvec;
    rvec.x = dot((float4)(mat[0],mat[4],mat[8],mat[12]),vec);
    rvec.y = dot((float4)(mat[1],mat[5],mat[9],mat[13]),vec);
    rvec.z = dot((float4)(mat[2],mat[6],mat[10],mat[14]),vec);
    rvec.w = dot((float4)(mat[3],mat[7],mat[11],mat[15]),vec);
    return rvec;
}

__kernel void project( __global float4* kin,
                    __global float4* color,
                     __global float* depth, 
                    __global uchar* rgb,    //packed as 3 floats from OpenCV
                     __constant float* pt, 
                     __constant float* ipt, 
                    int w,
                    int h)
{
    unsigned int i = get_global_id(0);
    
    int c = i % w;
    int r = (int)(i / w);
    //int r = i % h;
    //int c = (int)(i / w);



    float d = depth[i];
    int irgb = i*3;
    //irgb = r*(w*3) + c*3;
    float4 col = (float4)(rgb[irgb+2]/255.f, rgb[irgb+1]/255.f, rgb[irgb]/255.f, 1.0f);
    //col.y = 1;
    color[i] = col;
    //color[i] = (float4)(d/3., d/3., d/3., 1);
    //kin[i] = (float4)(d*c/w, d*r/h, d, 1);
    kin[i] = (float4)(1.5*c/w, 1.5*r/h, 0, 1);
    //color[i] = (float4)(1,0,0,0);
    //kin[i] = (float4)(0,0,0,0);
#if 1
    //unproject from depth, in place
    float4 epix = (float4)(d*c, d*r, d, 1);
    kin[i] = matmult4(ipt, epix);
    kin[i].w = 1;


    epix = matmult4(pt, kin[i]);
    int x = (int)(epix.x/epix.z);
    int y = (int)(epix.y/epix.z);
 
/*
   inline bool is_yx_in_range(const cv::Mat& image, int y, int x)
{ return (x >= 0) && (y >= 0) && (x < image.cols) && (y < image.rows); }
*/
    //if y,x in range

    if(x>=0 && y>=0 && x < w && y < h)
    {
        irgb = y*w*3 + x*3;
        color[i].x = rgb[irgb+2]/255.;
        color[i].y = rgb[irgb+1]/255.;
        color[i].z = rgb[irgb]/255.;
        //color[i] = (float4)(0,1,0,1);
    }
    else
    {
        color[i] = (float4)(0,0,1,1);
    }

#endif
}






#if 0
void Pose3D :: unprojectFromImage(const cv::Mat1f& pixels, const cv::Mat1b& mask, cv::Mat3f& voxels) const
{
  Eigen::Vector4d epix;
  Eigen::Vector4d evox;

  epix(3) = 1; // w does not change.

  for (int r = 0; r < pixels.rows; ++r)
  {
    const float* pixels_data = pixels.ptr<float>(r);
    const uchar* mask_data = mask.ptr<uchar>(r);
    Vec3f* voxels_data = voxels.ptr<Vec3f>(r);
    for (int c = 0; c < pixels.cols; ++c)
    {
      if (!mask[c])
        continue;
      const float d = pixels_data[c];
      epix(0) = c*d;
      epix(1) = r*d;
      epix(2) = d;
      evox = impl->inv_project_transform * epix;
      voxels_data[c][0] = evox(0);
      voxels_data[c][1] = evox(1);
      voxels_data[c][2] = evox(2);
    }
  }
}



void Pose3D :: projectToImage(const cv::Mat3f& voxels, const cv::Mat1b& mask, cv::Mat3f& pixels) const
{
  Eigen::Vector4d epix;
  Eigen::Vector4d evox;
  evox(3) = 1; // w does not change.

  for (int r = 0; r < voxels.rows; ++r)
  {
    const Vec3f* voxels_data = voxels.ptr<Vec3f>(r);
    const uchar* mask_data = mask.ptr<uchar>(r);
    Vec3f* pixels_data = pixels.ptr<Vec3f>(r);
    for (int c = 0; c < voxels.cols; ++c)
    {
      if (!mask[c])
        continue;
      evox(0) = voxels_data[c][0];
      evox(1) = voxels_data[c][1];
      evox(2) = voxels_data[c][2];
      epix = impl->project_transform * evox;
      pixels_data[c][0] = epix(0)/epix(2);
      pixels_data[c][1] = epix(1)/epix(2);
      pixels_data[c][2] = epix(2);
    }
  }
}




#endif
