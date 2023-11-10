function TestPosture()%------------one frame
setup() ;
%addpath('E:\old matconvnet');
%imageMean=0.0624;
filelist=[];filelist1=[];
net1=load('E:\old matconvnet\data\Posture\net-epoch-30.mat');%Leapscartch1%LeapAlex1%LeapFace1%LeapVGG-f%LeapVGG-s%Leapvgg-f

net1.net.layers(end)=[];%Remove the last layer
Tpath= {'E:\Matlab\Hand Posture and Gesture\Test'} ;

conf=('E:\Matlab\Hand Posture and Gesture\Posture');
mkdir(conf);

filelist=dir(Tpath{1});%Get the data from current dir
ER=0;c=0;Ja=0;Ja1=0;avJ=0;labels=0;
ff=0;rr=0;acc=0;ffd=0;rrd=0;accd=0;pred2v=[];ground2v=[];ground2=[];pred2=[];
%pathT='D:\chalearn\Python\gt';%%%%one
for gg=3:numel(filelist)%No of samples(sequences)
    seq=[];seq1=[];Array1=0;
    %scou=0;%%%%one
    file=filelist(gg);
    video11=fullfile(Tpath,file.name);
    
    
       
         L=(file.name(end-5));
      switch  L
      case 'a' 
        labels=1;
      case 'b' 
        labels=2;
      case 'c'
         labels=3;
      case 'd'
         labels=4;
      case 'g'
         labels=5;
      case 'h'
         labels=6;
      case 'i'
         labels=7;
      case 'v'
         labels=8;
      case 'y'
         labels=9;
      case 'l'
         labels=10;
      end 
    
  
        
        
       
        
        
        
        
        xx1=imread(video11{1});
        xx2 = imresize(xx1,[227 227]) ;
        data.imm7(:,:,:)=im2single(cat(3,xx2(:,:,:),xx2(:,:,:),xx2(:,:,:))); %%%
        %data.imm7=data.imm7+0.5062;
        res=vl_simplenn1(net1.net,data.imm7);
        scores=squeeze(gather(res(end).x));
        [score,pred]=max(scores);
        
        %%%%%%%%per frame accuracy
        %temporaly
        pred2(end+1)=pred;
        ground2(end+1)=labels;
        
        
   
end%epoch
%     F1=fullfile(pathT,file.name);
%     F11=strcat(F1,'_data.csv');
%     xlswrite(F11,scou,'A1:A1');
%     csvwrite(pvideo1,Array)
%     csvwrite(Gvideo1,Array1)
%     csvwrite(Gvideo2,Array1(size(Array1,1),3))
gg
 wrong=0;numRight=0;
for ii=1:length(pred2)
    if (pred2(ii) == ground2(ii))
        numRight = numRight + 1;
    else
        wrong = wrong+1;
    end
end
acc=(numRight/(numRight+wrong))*100
save(strcat(conf,'/pr.mat'),'pred2');
save(strcat(conf,'/gt.mat'),'ground2');

%%%%%%%
load(strcat(conf,'/pr.mat'));
load(strcat(conf,'/gt.mat'));

%%%

conver(ground2,pred2);

function conver(gt,p)
gt1 = zeros(10,length(gt),'double') ;
p1 = zeros(10,length(gt),'double') ;

for j = 1 : length(gt)
    a = gt(1,j) ;
    gt1(a,j) = 1 ;
end

for j = 1 : length(gt)
    a = p(1,j)  ;
    p1(a,j) = 1 ;
end
plotconfusion(gt1,p1);
end
end