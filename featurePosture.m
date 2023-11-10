
function featurePosture()
setup() ;
%addpath('E:\old matconvnet');
%imageMean=0.0624;
filelist=[];filelist1=[];cou=1;

Tpath= {'E:\Matlab\Hand Posture and Gesture\Test'} ;
load('E:\old matconvnet\data\PostureFace\net-epoch-30.mat');
net.layers(end)=[];
filelist=dir(Tpath{1});%Get the data from current dir
label=[];
im=[];
gg=1;
%pathT='D:\chalearn\Python\gt';%%%%one
for epoch=3:numel(filelist)%No of samples(sequences)
    seq=[];seq1=[];Array1=0;
    %scou=0;%%%%one
    file=filelist(epoch);
    video11=file.name;
    
    
    
        video1=fullfile(Tpath,video11);
        
        
        
        pred22=[];
        
         L=(video11(end-5));
      switch  L
      case 'a' 
        label(end+1)=1;
      case 'b' 
        label(end+1)=2;
      case 'c'
         label(end+1)=3;
      case 'd'
         label(end+1)=4;
      case 'g'
         label(end+1)=5;
      case 'h'
         label(end+1)=6;
      case 'i'
         label(end+1)=7;
      case 'v'
         label(end+1)=8;
      case 'y'
         label(end+1)=9;
      case 'l'
         label(end+1)=10;
      end 
        
        
        
        
        xx1=imread(video1{1});
        xx2 = im2single(imresize(xx1,[224 224])) ;
        xx2=cat(3,xx2,xx2,xx2);
        res=vl_simplenn1(net,xx2);
        
        d=squeeze(gather(res(end-2).x)); %%%
        try
        im=[im,d];
        
        catch
         label
        end
        
        
   
    
   
   
    
end%epoch
mkdir('E:\Matlab\PostureGesture');
save('E:\Matlab\PostureGesture\Testfeature.mat','im','label');
