%this is the main program which requires input file initialDes.mat file and
%another input file of values of DOE of vf and rmin vfrmin.mat along with 
%subroutine Top3D_maxshear.m. It generates output file shear.txt. Save the
%iteration details to a output file dataGenertion32.out
penal=5;%penalisation factor
ft=2;%sensitivity filtering during optimization
disp('loading input')
inp_param=load ('vfrmin.mat');
inp_des=load ('initialDes.mat');
voxel=inp_des.voxel;
volfraction=inp_param.var(:,1);
filter=inp_param.var(:,2);
SHEAR=1;
for ii=1:2751
    try
        [density,Q,iter] = Top3D_maxshear([0.01 0.01 0.01 32 32 32],voxel, volfraction(ii), penal, filter(ii),ft);
        if iter<1000
            row     = [SHEAR,volfraction(ii), filter(ii), density(:)',Q(:)'];
            matrix(ii,:) = row;
        end
    catch
    end
end
save('shear.txt', 'matrix', '-ascii', '-tabs')

