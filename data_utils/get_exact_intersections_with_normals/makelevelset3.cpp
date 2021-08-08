#include "makelevelset3.h"

// find distance x0 is from segment x1-x2
static float point_segment_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2)
{
   Vec3f dx(x2-x1);
   double m2=mag2(dx);
   // find parameter value of closest point on segment
   float s12=(float)(dot(x2-x0, dx)/m2);
   if(s12<0){
      s12=0;
   }else if(s12>1){
      s12=1;
   }
   // and find the distance
   return dist(x0, s12*x1+(1-s12)*x2);
}

// find distance x0 is from triangle x1-x2-x3
static float point_triangle_distance(const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, const Vec3f &x3)
{
   // first find barycentric coordinates of closest point on infinite plane
   Vec3f x13(x1-x3), x23(x2-x3), x03(x0-x3);
   float m13=mag2(x13), m23=mag2(x23), d=dot(x13,x23);
   float invdet=1.f/max(m13*m23-d*d,1e-30f);
   float a=dot(x13,x03), b=dot(x23,x03);
   // the barycentric coordinates themselves
   float w23=invdet*(m23*a-d*b);
   float w31=invdet*(m13*b-d*a);
   float w12=1-w23-w31;
   if(w23>=0 && w31>=0 && w12>=0){ // if we're inside the triangle
      return dist(x0, w23*x1+w31*x2+w12*x3); 
   }else{ // we have to clamp to one of the edges
      if(w23>0) // this rules out edge 2-3 for us
         return min(point_segment_distance(x0,x1,x2), point_segment_distance(x0,x1,x3));
      else if(w31>0) // this rules out edge 1-3
         return min(point_segment_distance(x0,x1,x2), point_segment_distance(x0,x2,x3));
      else // w12 must be >0, ruling out edge 1-2
         return min(point_segment_distance(x0,x1,x3), point_segment_distance(x0,x2,x3));
   }
}

// calculate twice signed area of triangle (0,0)-(x1,y1)-(x2,y2)
// return an SOS-determined sign (-1, +1, or 0 only if it's a truly degenerate triangle)
static int orientation(double x1, double y1, double x2, double y2, double &twice_signed_area)
{
   twice_signed_area=y1*x2-x1*y2;
   if(twice_signed_area>0) return 1;
   else if(twice_signed_area<0) return -1;
   else if(y2>y1) return 1;
   else if(y2<y1) return -1;
   else if(x1>x2) return 1;
   else if(x1<x2) return -1;
   else return 0; // only true when x1==x2 and y1==y2
}

// robust test of (x0,y0) in the triangle (x1,y1)-(x2,y2)-(x3,y3)
// if true is returned, the barycentric coordinates are set in a,b,c.
static bool point_in_triangle_2d(double x0, double y0, 
                                 double x1, double y1, double x2, double y2, double x3, double y3,
                                 double& a, double& b, double& c)
{
   x1-=x0; x2-=x0; x3-=x0;
   y1-=y0; y2-=y0; y3-=y0;
   int signa=orientation(x2, y2, x3, y3, a);
   if(signa==0) return false;
   int signb=orientation(x3, y3, x1, y1, b);
   if(signb!=signa) return false;
   int signc=orientation(x1, y1, x2, y2, c);
   if(signc!=signa) return false;
   double sum=a+b+c;
   assert(sum!=0); // if the SOS signs match and are nonkero, there's no way all of a, b, and c are zero.
   a/=sum;
   b/=sum;
   c/=sum;
   return true;
}

void make_level_set3(const std::vector<Vec3ui> &tri, const std::vector<Vec3f> &x,
                     const Vec3f &origin, float dx, int ni, int nj, int nk,
                     int &phi_X_N, int &phi_Y_N, int &phi_Z_N,
                     Array2f &phi_X, Array2f &phi_Y, Array2f &phi_Z,
                     const int exact_band)
{
   phi_X.resize((ni*ni)*16, 6); phi_X.assign(0);
   phi_Y.resize((ni*ni)*16, 6); phi_Y.assign(0);
   phi_Z.resize((ni*ni)*16, 6); phi_Z.assign(0);
   phi_X_N = 0;
   phi_Y_N = 0;
   phi_Z_N = 0;

   for(unsigned int t=0; t<tri.size(); ++t){
      unsigned int p, q, r; assign(tri[t], p, q, r);
      // coordinates in grid to high precision
      double fip=((double)x[p][0]-origin[0])/dx, fjp=((double)x[p][1]-origin[1])/dx, fkp=((double)x[p][2]-origin[2])/dx;
      double fiq=((double)x[q][0]-origin[0])/dx, fjq=((double)x[q][1]-origin[1])/dx, fkq=((double)x[q][2]-origin[2])/dx;
      double fir=((double)x[r][0]-origin[0])/dx, fjr=((double)x[r][1]-origin[1])/dx, fkr=((double)x[r][2]-origin[2])/dx;

      double a,b,c,x,y,z,ti,tj,tk,area,ip;
      int i0,i1,j0,j1,k0,k1;

      //compute normal
      a = fiq - fip;
      b = fjq - fjp;
      c = fkq - fkp;
      x = fir - fip;
      y = fjr - fjp;
      z = fkr - fkp;
      ti = b*z-c*y;
      tj = c*x-a*z;
      tk = a*y-b*x;
      area = sqrt(ti*ti+tj*tj+tk*tk);
      if (area>1e-30){
         ti = ti/area;
         tj = tj/area;
         tk = tk/area;
      }
      else {
         ti = 0;
         tj = 0;
         tk = 0;
      }

      // intersections -- i coordinate
      j0=(int)std::ceil(min(fjp,fjq,fjr));
      j1=(int)std::floor(max(fjp,fjq,fjr));
      k0=(int)std::ceil(min(fkp,fkq,fkr));
      k1=(int)std::floor(max(fkp,fkq,fkr));
      for(int k=k0; k<=k1; ++k) for(int j=j0; j<=j1; ++j){
         if(point_in_triangle_2d(j, k, fjp, fkp, fjq, fkq, fjr, fkr, a, b, c)){
            ip = a*fip+b*fiq+c*fir;
            phi_X(phi_X_N,0) = ip;
            phi_X(phi_X_N,1) = j;
            phi_X(phi_X_N,2) = k;
            phi_X(phi_X_N,3) = ti;
            phi_X(phi_X_N,4) = tj;
            phi_X(phi_X_N,5) = tk;
            ++phi_X_N;
         }
      }

      // intersections -- j coordinate
      i0=(int)std::ceil(min(fip,fiq,fir));
      i1=(int)std::floor(max(fip,fiq,fir));
      k0=(int)std::ceil(min(fkp,fkq,fkr));
      k1=(int)std::floor(max(fkp,fkq,fkr));
      for(int k=k0; k<=k1; ++k) for(int i=i0; i<=i1; ++i){
         if(point_in_triangle_2d(i, k, fip, fkp, fiq, fkq, fir, fkr, a, b, c)){
            ip = a*fjp+b*fjq+c*fjr;
            phi_Y(phi_Y_N,0) = i;
            phi_Y(phi_Y_N,1) = ip;
            phi_Y(phi_Y_N,2) = k;
            phi_Y(phi_Y_N,3) = ti;
            phi_Y(phi_Y_N,4) = tj;
            phi_Y(phi_Y_N,5) = tk;
            ++phi_Y_N;
         }
      }

      // intersections -- k coordinate
      j0=(int)std::ceil(min(fjp,fjq,fjr));
      j1=(int)std::floor(max(fjp,fjq,fjr));
      i0=(int)std::ceil(min(fip,fiq,fir));
      i1=(int)std::floor(max(fip,fiq,fir));
      for(int i=i0; i<=i1; ++i) for(int j=j0; j<=j1; ++j){
         if(point_in_triangle_2d(j, i, fjp, fip, fjq, fiq, fjr, fir, a, b, c)){
            ip = a*fkp+b*fkq+c*fkr;
            phi_Z(phi_Z_N,0) = i;
            phi_Z(phi_Z_N,1) = j;
            phi_Z(phi_Z_N,2) = ip;
            phi_Z(phi_Z_N,3) = ti;
            phi_Z(phi_Z_N,4) = tj;
            phi_Z(phi_Z_N,5) = tk;
            ++phi_Z_N;
         }
      }

   }
}

