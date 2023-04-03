n=10;%mesh size
[xi,yi,zi]= meshgrid(0:1/(n-1):1,0:1/(n-1):1,0:1/(n-1):1);L=1;c=0;
gyr = sin(2*pi*xi/L).*cos(2*pi*yi/L) + sin(2*pi*yi/L).*cos(2*pi*zi/L) + sin(2*pi*zi/L).*cos(2*pi*xi/L) -c;
prim = cos(2*pi*xi/L) + cos(2*pi*yi/L) + cos(2*pi*zi/L) - c;
diam = cos(2*pi*xi/L).*cos(2*pi*yi/L).*cos(2*pi*zi/L) -sin(2*pi*xi/L).*sin(2*pi*yi/L).*sin(2*pi*zi/L) - c;
fv=isosurface(xi,yi,zi,gyr);
ind=repmat(fv.faces,1,2);
nind=ind(:,[1 2 5 3 6 4]);
struts=reshape(nind',2,numel(nind)/2);
struts=[1:size(struts,2); struts];
verticesfw=[1:size(fv.vertices,1); fv.vertices'];
fileID = fopen('gyroid.txt','w');
formatSpec = 'GRID %4d %14.4f %7.4f %7.4f\n';
fprintf(fileID,formatSpec,verticesfw);
formatSpec = 'STRUT %5d %12d %7d        \n';
fprintf(fileID,formatSpec,struts);
fclose(fileID);
[voxel,Density] = GenerateVoxel(40,'gyroid.txt',0.1);
display_3D(voxel)