function display_3D(rho)
[nely,nelx,nelz] = size(rho);
hx = 1; hy = 1; hz = 1;
face = [1 2 3 4; 2 6 7 3; 4 3 7 8; 1 5 8 4; 1 2 6 5; 5 6 7 8];
set(gcf,'Name','ISO display','NumberTitle','off');
for k = 1:nelz
    z = (k-1)*hz;
    for i = 1:nelx
        x = (i-1)*hx;
        %count =0;
        for j = 1:nely
            y = nely*hy - (j-1)*hy;
             if (rho(j,i,k) > 0.5)
                vert = [x y z; x y-hx z; x+hx y-hx z; x+hx y z; x y z+hx;x y-hx z+hx; x+hx y-hx z+hx;x+hx y z+hx];
                vert(:,[2 3]) = vert(:,[3 2]); vert(:,2,:) = -vert(:,2,:);
                patch('Faces',face,'Vertices',vert,'FaceColor','red');%[0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k)),0.2+0.8*(1-rho(j,i,k))]);
%                 if(count<3)
%                 patch('Faces',face,'Vertices',vert,'FaceColor',[1,0,0],'EdgeColor',[0,0,0]);
%                 text(x+0.5, -y-0.1, z+0.5, num2str(1),'HorizontalAlignment','center','VerticalAlignment','middle','FontUnits', 'Normalized', 'FontSize', 0.08);
%                 else
%                 patch('Faces',face,'Vertices',vert,'FaceColor',[1,1,1],'EdgeColor',[0,0,0]);
%                 text(x+0.5, -y-0.1, z+0.5, num2str(0),'HorizontalAlignment','center','VerticalAlignment','middle','FontUnits', 'Normalized', 'FontSize', 0.08);
%                 end
%                 hold on;
%                 count=count+1;
            end
        end
    end
end
axis equal; axis tight; axis off; box on; view([30,30]); pause(1e-6);
end