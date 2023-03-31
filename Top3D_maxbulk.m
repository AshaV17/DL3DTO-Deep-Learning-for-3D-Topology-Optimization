function [xPhys,DH,loop]=Top3D_maxbulk(Micro_struct,voxel, volfrac, penal, rmin,ft)
%Here, the voxel data of gyroid is used as initial design for TO
% USER-DEFINED LOOP PARAMETERS
maxloop = 600; displayflag = 0; 
%MATERIAL PROPERTIES
E0 = 1; Emin = 1e-9; nu = 0.3;
%% PREPARE FINITE ELEMENT ANALYSIS
%Ke= lk_H8(nu);
lx = Micro_struct(1); ly = Micro_struct(2); lz = Micro_struct(3);
nelx   = Micro_struct(4); nely  = Micro_struct(5); nelz   = Micro_struct(6);
%ndof = 3*(nelx+1)*(nely+1)*(nelz+1);
%U = zeros(ndof,1); 
%%ISOPARAMETRIC K ELEMENT STIFFNESS MATRIX
D0 = E0/(1+nu)/(1-2*nu)*...
    [ 1-nu   nu   nu     0          0          0     ;
        nu 1-nu   nu     0          0          0     ;
        nu   nu 1-nu     0          0          0     ;
         0    0    0 (1-2*nu)/2     0          0     ;
         0    0    0     0      (1-2*nu)/2     0     ;
         0    0    0     0          0      (1-2*nu)/2];
 nele = nelx*nely*nelz;
 dx = lx/nelx; dy = ly/nely; dz = lz/nelz;
 Ke = elementMatVec3D(dx/2, dy/2, dz/2, D0);
Num_node = (1+nely)*(1+nelx)*(1+nelz);
nodenrs = reshape(1:Num_node,1+nely,1+nelx,1+nelz);
edofVec = reshape(3*nodenrs(1:end-1,1:end-1,1:end-1)+1,nelx*nely*nelz,1);
edofMat = repmat(edofVec,1,24)+repmat([0 1 2 3*nely+[3 4 5 0 1 2] -3 -2 -1 ...
    3*(nelx+1)*(nely+1)+[0 1 2 3*nely+[3 4 5 0 1 2] -3 -2 -1]], nele, 1);
iK = reshape(kron(edofMat,ones(24,1))',24*24*nele,1);
jK = reshape(kron(edofMat,ones(1,24))',24*24*nele,1);
% PREPARE FILTER
[Micro.H,Micro.Hs] =  filtering3d(nelx, nely, nelz, nele, rmin);
% 3D periodic boundary formulation
% the nodes classification
n1 = [nodenrs(end, [1 end], 1) nodenrs(1, [end 1], 1) nodenrs(end, [1 end], end) nodenrs(1, [end 1], end)];
d1 = reshape([3*n1-2; 3*n1-1; 3*n1],3*numel(n1),1);
n3 = [reshape(squeeze(nodenrs(end,1,2:end-1)),1,numel(squeeze(nodenrs(end,1,2:end-1))))...              % AE
      reshape(squeeze(nodenrs(1, 1, 2:end-1)),1,numel(squeeze(nodenrs(1, 1, 2:end-1))))...              % DH
      reshape(squeeze(nodenrs(end,2:end-1,1)),1,numel(squeeze(nodenrs(end,2:end-1,1))))...              % AB
      reshape(squeeze(nodenrs(1, 2:end-1, 1)),1,numel(squeeze(nodenrs(1, 2:end-1, 1))))...              % DC
      reshape(squeeze(nodenrs(2:end-1, 1, 1)),1,numel(squeeze(nodenrs(2:end-1, 1, 1))))...              % AD
      reshape(squeeze(nodenrs(2:end-1,1,end)),1,numel(squeeze(nodenrs(2:end-1,1,end))))...              % EH
      reshape(squeeze(nodenrs(2:end-1, 2:end-1, 1)),1,numel(squeeze(nodenrs(2:end-1, 2:end-1, 1))))...  % ABCD
      reshape(squeeze(nodenrs(2:end-1, 1, 2:end-1)),1,numel(squeeze(nodenrs(2:end-1, 1, 2:end-1))))...  % ADHE
      reshape(squeeze(nodenrs(end,2:end-1,2:end-1)),1,numel(squeeze(nodenrs(end,2:end-1,2:end-1))))];   % ABFE                   
d3 = reshape([3*n3-2; 3*n3-1; 3*n3],3*numel(n3),1);
n4 = [reshape(squeeze(nodenrs(1, end, 2:end-1)),1,numel(squeeze(nodenrs(1, end, 2:end-1))))...          % CG
      reshape(squeeze(nodenrs(end,end,2:end-1)),1,numel(squeeze(nodenrs(end,end,2:end-1))))...          % BF
      reshape(squeeze(nodenrs(1, 2:end-1, end)),1,numel(squeeze(nodenrs(1, 2:end-1, end))))...          % HG
      reshape(squeeze(nodenrs(end,2:end-1,end)),1,numel(squeeze(nodenrs(end,2:end-1,end))))...          % EF
      reshape(squeeze(nodenrs(2:end-1,end,end)),1,numel(squeeze(nodenrs(2:end-1,end,end))))...          % FG
      reshape(squeeze(nodenrs(2:end-1, end, 1)),1,numel(squeeze(nodenrs(2:end-1, end, 1))))...          % BC
      reshape(squeeze(nodenrs(2:end-1,2:end-1,end)),1,numel(squeeze(nodenrs(2:end-1,2:end-1,end))))...  % EFGH
      reshape(squeeze(nodenrs(2:end-1,end,2:end-1)),1,numel(squeeze(nodenrs(2:end-1,end,2:end-1))))...  % BCGF
      reshape(squeeze(nodenrs(1, 2:end-1, 2:end-1)),1,numel(squeeze(nodenrs(1, 2:end-1, 2:end-1))))];   % DCGH
d4 = reshape([3*n4-2; 3*n4-1; 3*n4],3*numel(n4),1);
n2 = setdiff(nodenrs(:),[n1(:);n3(:);n4(:)]); d2 = reshape([3*n2-2; 3*n2-1; 3*n2],3*numel(n2),1);
% the imposing of six linearly independent unit test strains
e = eye(6); ufixed = zeros(24,6);
vert_cor = [0, lx, lx,  0,  0, lx, lx,  0;
            0,  0, ly, ly,  0,  0, ly, ly;
            0,  0,  0,  0, lz, lz, lz, lz];
for i = 1:6
    epsilon = [  e(i,1), e(i,4)/2, e(i,6)/2;
               e(i,4)/2,   e(i,2), e(i,5)/2;
               e(i,6)/2, e(i,5)/2,   e(i,3)];
    ufixed(:,i) = reshape(epsilon*vert_cor,24,1);
end
% 3D boundary constraint equations
wfixed = [repmat(ufixed(  7:9,:),numel(squeeze(nodenrs(end,1,2:end-1))),1);                    % C
          repmat(ufixed(  4:6,:)-ufixed(10:12,:),numel(squeeze(nodenrs(1, 1, 2:end-1))),1);    % B-D
          repmat(ufixed(22:24,:),numel(squeeze(nodenrs(end,2:end-1,1))),1);                    % H
          repmat(ufixed(13:15,:)-ufixed(10:12,:),numel(squeeze(nodenrs(1, 2:end-1, 1))),1);    % E-D
          repmat(ufixed(16:18,:),numel(squeeze(nodenrs(2:end-1, 1, 1))),1);                    % F
          repmat(ufixed(  4:6,:)-ufixed(13:15,:),numel(squeeze(nodenrs(2:end-1,1,end))),1);    % B-E
          repmat(ufixed(13:15,:),numel(squeeze(nodenrs(2:end-1, 2:end-1, 1))),1);              % E
          repmat(ufixed(  4:6,:),numel(squeeze(nodenrs(2:end-1, 1, 2:end-1))),1);              % B
          repmat(ufixed(10:12,:),numel(squeeze(nodenrs(end,2:end-1,2:end-1))),1)];             % D
% INITIALIZE ITERATION
qe = cell(6,6);DH = zeros(6,6); dDH = cell(6,6);
cellVolume = lx*ly*lz; 
Micro.x = voxel;
%voids=(voxel==0);
%  Micro.x = repmat(volfrac,[nely,nelx,nelz]);%ones(nely,nelx,nelz);
%  Micro.x(nely/2:nely/2+3,nelx/2:nelx/2+3,1:nelz) = 0;%to put a hole in initial design
%beta = 1; 
%Micro.xTilde = Micro.x;
xPhys =Micro.x;
%Micro.xPhys = 1-exp(-beta*Micro.xTilde)+Micro.xTilde*exp(-beta);%Heaveside fn
loop = 0;  Micro.change = 1;
while Micro.change > 0.01
    loop = loop+1; 
    if loop>maxloop 
        break; 
    end
    %loopbeta = loopbeta+1;
    % the reduced elastic equilibrium equation to compute the induced displacement field
    sK = reshape(Ke(:)*(Emin+xPhys(:)'.^penal*(1-Emin)),24*24*nele,1);
    K = sparse(iK(:), jK(:), sK(:)); K = (K+K')/2;
    Kr = [K(d2,d2), K(d2,d3)+K(d2,d4); K(d3,d2)+K(d4,d2),K(d3,d3)+K(d4,d3)+K(d3,d4)+K(d4,d4)];
    U(d1,:)= ufixed;
    U([d2;d3],:) = Kr\(-[K(d2,d1);K(d3,d1)+K(d4,d1)]*ufixed-[K(d2,d4);K(d3,d4)+K(d4,d4)]*wfixed);
    U(d4,:) = U(d3,:) + wfixed;
    % homogenization to evaluate macroscopic effective properties
    for i = 1:6
        for j = 1:6
            U1 = U(:,i); U2 = U(:,j);
            qe{i,j} = reshape(sum((U1(edofMat)*Ke).*U2(edofMat),2),nely,nelx,nelz);
            DH(i,j) = 1/cellVolume*sum(sum(sum((Emin+xPhys.^penal*(E0-Emin)).*qe{i,j})));
            dDH{i,j} = 1/cellVolume*(penal*(E0-Emin)*xPhys.^(penal-1).*qe{i,j});
        end
    end
    %disp('--- Homogenized elasticity tensor ---'); disp(DH)
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    c=-(DH(1,1)+DH(2,2)+DH(3,3)+DH(1,2)+DH(1,3)+DH(2,1)+DH(3,1)+DH(2,3)+DH(3,2));%maximizing bulk modulus
    dc = -(dDH{1,1}+dDH{2,2}+dDH{3,3}+dDH{1,2}+dDH{1,3}+dDH{2,1}+dDH{3,1}+dDH{2,3}+dDH{3,2});
    dv = ones(nely, nelx, nelz);
    % FILTERING AND MODIFICATION OF SENSITIVITIES 
    if ft==2 
        dc(:) = Micro.H*(dc(:).*dx(:)./Micro.Hs); 
        dv(:) = Micro.H*(dv(:).*dx(:)./Micro.Hs); 
    end
    % OPTIMALITY CRITERIA UPDATE 
    [Micro.x, xPhys, Micro.change] = OC(Micro.x, dc, dv, Micro.H, Micro.Hs, volfrac, nele, 0.2,0);%, beta);
    xPhys = reshape(xPhys, nely, nelx, nelz);  
    %Micro.xPhys(voids)=0;
    % PRINT RESULTS
    fprintf(' It.:%5i Obj.:%11.4f Micro_Vol.:%7.3f Micro_ch.:%7.3f\n',...
        loop,c,mean(xPhys(:)),Micro.change);
    if displayflag, clf; display_3D(xPhys); end
%     % UPDATE HEAVISIDE REGULARIZATION PARAMETER
%     if beta < 512 && (loopbeta >= 50 ||  Micro.change <= 0.01 )
%         beta = 2*beta; loopbeta = 0; Micro.change = 1;
%         fprintf('Parameter beta increased to %g.\n',beta);
%     end
end
end
%% SUB FUNCTION:filtering3D
function [H,Hs] = filtering3d(nelx, nely, nelz, nele, rmin)
iH = ones(nele*(2*(ceil(rmin)-1)+1)^2,1);
jH = ones(size(iH));
sH = zeros(size(iH)); 
k = 0;
for k1 = 1:nelz
    for i1 = 1:nelx
        for j1 = 1:nely
            e1 = (k1-1)*nelx*nely + (i1-1)*nely+j1;
            for k2 = max(k1-(ceil(rmin)-1),1):min(k1+(ceil(rmin)-1),nelz)
                for i2 = max(i1-(ceil(rmin)-1),1):min(i1+(ceil(rmin)-1),nelx)
                    for j2 = max(j1-(ceil(rmin)-1),1):min(j1+(ceil(rmin)-1),nely)
                        e2 = (k2-1)*nelx*nely + (i2-1)*nely+j2;
                        k = k+1;
                        iH(k) = e1;
                        jH(k) = e2;
                        sH(k) = max(0,rmin-sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2));
                    end
                end
            end
        end
    end
end
H = sparse(iH,jH,sH); Hs = sum(H,2);
end
%% SUB FUNCTION: elementMatVec3D
function Ke = elementMatVec3D(a, b, c, DH)
GN_x=[-1/sqrt(3),1/sqrt(3)]; GN_y=GN_x; GN_z=GN_x; GaussWeigh=[1,1];
Ke = zeros(24,24); L = zeros(6,9);
L(1,1) = 1; L(2,5) = 1; L(3,9) = 1;
L(4,2) = 1; L(4,4) = 1; L(5,6) = 1;
L(5,8) = 1; L(6,3) = 1; L(6,7) = 1;
for ii=1:length(GN_x)
    for jj=1:length(GN_y)
        for kk=1:length(GN_z)
            x = GN_x(ii);y = GN_y(jj);z = GN_z(kk);
            dNx = 1/8*[-(1-y)*(1-z)  (1-y)*(1-z)  (1+y)*(1-z) -(1+y)*(1-z) -(1-y)*(1+z)  (1-y)*(1+z)  (1+y)*(1+z) -(1+y)*(1+z)];
            dNy = 1/8*[-(1-x)*(1-z) -(1+x)*(1-z)  (1+x)*(1-z)  (1-x)*(1-z) -(1-x)*(1+z) -(1+x)*(1+z)  (1+x)*(1+z)  (1-x)*(1+z)];
            dNz = 1/8*[-(1-x)*(1-y) -(1+x)*(1-y) -(1+x)*(1+y) -(1-x)*(1+y)  (1-x)*(1-y)  (1+x)*(1-y)  (1+x)*(1+y)  (1-x)*(1+y)];
            J = [dNx;dNy;dNz]*[ -a  a  a  -a  -a  a  a  -a ;  -b  -b  b  b  -b  -b  b  b; -c -c -c -c  c  c  c  c]';
            G = [inv(J) zeros(3) zeros(3);zeros(3) inv(J) zeros(3);zeros(3) zeros(3) inv(J)];
            dN(1,1:3:24) = dNx; dN(2,1:3:24) = dNy; dN(3,1:3:24) = dNz;
            dN(4,2:3:24) = dNx; dN(5,2:3:24) = dNy; dN(6,2:3:24) = dNz;
            dN(7,3:3:24) = dNx; dN(8,3:3:24) = dNy; dN(9,3:3:24) = dNz;
            Be = L*G*dN;
            Ke = Ke + GaussWeigh(ii)*GaussWeigh(jj)*GaussWeigh(kk)*det(J)*(Be'*DH*Be);
        end
    end
end
end
%% SUB FUNCTION: OC
function [x, xPhys, change] = OC(x, dc, dv, H, Hs, volfrac, nele, move, beta)
l1 = 0; l2 = 1e9;
while (l2-l1)/(l1+l2) > 1e-3
    lmid = 0.5*(l2+l1);
    xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
    xPhys(:) = (H*xnew(:))./Hs; 
    %xTilde(:) = (H*xnew(:))./Hs; xPhys =x;%xPhys = 1-exp(-beta*xTilde)+xTilde*exp(-beta);
    if sum(xPhys(:)) > volfrac*nele, l1 = lmid; else, l2 = lmid; end
end
change = max(abs(xnew(:)-x(:))); x = xnew;
end
%% SUB FUNCTION: display_3D
function display_3D(rho)
[nely,nelx,nelz] = size(rho);
hx = 1; hy = 1; hz = 1;
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'Name','ISO display','NumberTitle','off');
for k = 1:nelz
    z = (k-1)*hz;
    for i = 1:nelx
        x = (i-1)*hx;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
            if (rho(j,i,k) > 0.5)
                vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'FaceColor',[0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k))]);
                hold on;
            end
        end
    end
end
axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6);
end
%======================================================================================================================%
% Function ConTop3D:                                                                                                   %
% A compact and efficient MATLAB code for Concurrent topology optimization of multiscale composite structures          %
% in Matlab.                                                                                                           %
%                                                                                                                      %
% Developed by: Jie Gao, Zhen Luo, Liang Xia and Liang Gao*                                                            %
% Email: gaoliang@mail.hust.edu.cn (GabrielJie_Tian@163.com)                                                           %
%                                                                                                                      %
% Main references:                                                                                                     %
%                                                                                                                      %
% (1) Jie Gao, Zhen Luo, Liang Xia, Liang Gao. Concurrent topology optimization of multiscale composite structures     %
% in Matlab. Accepted in Structural and multidisciplinary optimization.                                                %
%                                                                                                                      %
% (2) Xia L, Breitkopf P. Design of materials using topology optimization and energy-based homogenization approach in  %
% Matlab. % Structural and multidisciplinary optimization, 2015, 52(6): 1229-1241.                                     %
%                                                                                                                      %
% *********************************************   Disclaimer   ******************************************************* %
% The authors reserve all rights for the programs. The programs may be distributed and used for academic and           %
% educational purposes. The authors do not guarantee that the code is free from errors,and they shall not be liable    %
% in any event caused by the use of the program.                                                                       %
%======================================================================================================================%
