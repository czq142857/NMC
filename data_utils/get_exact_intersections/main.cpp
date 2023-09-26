//SDFGen - A simple grid-based signed distance field (level set) generator for triangle meshes.
//Written by Christopher Batty (christopherbatty@yahoo.com, www.cs.columbia.edu/~batty)
//...primarily using code from Robert Bridson's website (www.cs.ubc.ca/~rbridson)
//This code is public domain. Feel free to mess with it, let me know if you like it.

#include "makelevelset3.h"
#include "config.h"

#ifdef HAVE_VTK
  #include <vtkImageData.h>
  #include <vtkFloatArray.h>
  #include <vtkXMLImageDataWriter.h>
  #include <vtkPointData.h>
  #include <vtkSmartPointer.h>
#endif


#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>

int main(int argc, char* argv[]) {
  
  if(argc != 4) {
    std::cout << "IntersectionXYZ - A utility for sampling points on triangle meshes.\n";
    std::cout << "\nThe sampled points are intersection points between the mesh and the edges of a uniform grid.\n";
    std::cout << "The output filename will match that of the input, with the OBJ suffix replaced with .intersection.\n\n";

    std::cout << "Usage: IntersectionXYZ <filename> <size> <padding>\n\n";
    std::cout << "Where:\n";
    std::cout << "\t<filename> specifies a Wavefront OBJ (text) file representing a *triangle* mesh (no quad or poly meshes allowed). The shape is assumed to be normalized to [-0.5,0.5]. File must use the suffix \".obj\".\n";
    std::cout << "\t<size> specifies the number of grid cells in [-0.5,0.5].\n";
    std::cout << "\t<padding> specifies the number of cells worth of padding outside the [-0.5,0.5] bound box.\n\n";
    
    exit(-1);
  }

  std::string filename(argv[1]);
  if(filename.size() < 5 || filename.substr(filename.size()-4) != std::string(".obj")) {
    std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
    exit(-1);
  }

  std::stringstream arg2(argv[2]);
  int dsize;
  float dx;
  arg2 >> dsize;
  dx = 1.f/dsize;
  
  std::stringstream arg3(argv[3]);
  int padding;
  arg3 >> padding;

  //start with a massive inside out bound box.
  Vec3f min_box(0,0,0), max_box(0,0,0);
  
  //std::cout << "Reading data.\n";

  std::ifstream infile(argv[1]);
  if(!infile) {
    std::cerr << "Failed to open. Terminating.\n";
    exit(-1);
  }

  int ignored_lines = 0;
  std::string line;
  std::vector<Vec3f> vertList;
  std::vector<Vec3ui> faceList;
  while(!infile.eof()) {
    std::getline(infile, line);

    //.obj files sometimes contain vertex normals indicated by "vn"
    if(line.substr(0,2) == std::string("v ")){
      std::stringstream data(line);
      char c;
      Vec3f point;
      data >> c >> point[0] >> point[1] >> point[2];
      vertList.push_back(point);
    }
    else if(line.substr(0,2) == std::string("f ")) {
      std::stringstream data(line);
      std::string v0s,v1s,v2s;
      char c;
      int v0,v1,v2;
      data >> c >> v0s >> v1s >> v2s;
      v0 = stoi(v0s);
      v1 = stoi(v1s);
      v2 = stoi(v2s);
      faceList.push_back(Vec3ui(v0-1,v1-1,v2-1));
    }
    else {
      ++ignored_lines; 
    }
  }
  infile.close();
  
  //if(ignored_lines > 0)
  //  std::cout << "Warning: " << ignored_lines << " lines were ignored since they did not contain faces or vertices.\n";

  //std::cout << "Read in " << vertList.size() << " vertices and " << faceList.size() << " faces." << std::endl;

  //Add padding around the box.
  Vec3f unit(1,1,1);
  min_box -= (0.5f+padding*dx)*unit;
  max_box += (0.5f+dx+padding*dx)*unit;
  Vec3ui sizes = Vec3ui((max_box - min_box)/dx);
  
  //std::cout << "Bound box size: (" << min_box << ") to (" << max_box << ") with dimensions " << sizes << "." << std::endl;

  //std::cout << "Computing intersections.\n";

  int phi_X_N, phi_Y_N, phi_Z_N; // store number of sampled points
  Array2f phi_X, phi_Y, phi_Z; //Nx3 arrays to store sampled points

  make_level_set3(faceList, vertList, min_box, dx, sizes[0], sizes[1], sizes[2],
    phi_X_N, phi_Y_N, phi_Z_N,
    phi_X, phi_Y, phi_Z);

  std::string outname;


  outname = filename.substr(0, filename.size()-4) + std::string(".intersection");
  //std::cout << "Writing results to: " << outname << "\n";
  
  std::ofstream outfile( outname.c_str());
  outfile << "#intersection 1\n";
  outfile << "dim " << sizes[0] << " " << sizes[1] << " " << sizes[2] << "\n";
  outfile << phi_X_N << " " << phi_Y_N << " " << phi_Z_N << "\n";
  outfile << "data\n";

  char charv[4];
  float *floatv = reinterpret_cast<float*>(charv);

  for(int i=0; i<phi_X_N; ++i) for(int j=0; j<3; ++j) {
    *floatv = phi_X(i,j);
    outfile << charv[0] << charv[1] << charv[2] << charv[3];
  }
  for(int i=0; i<phi_Y_N; ++i) for(int j=0; j<3; ++j) {
    *floatv = phi_Y(i,j);
    outfile << charv[0] << charv[1] << charv[2] << charv[3];
  }
  for(int i=0; i<phi_Z_N; ++i) for(int j=0; j<3; ++j) {
    *floatv = phi_Z(i,j);
    outfile << charv[0] << charv[1] << charv[2] << charv[3];
  }

  outfile.close();


  //std::cout << "Processing complete.\n";

return 0;
}
