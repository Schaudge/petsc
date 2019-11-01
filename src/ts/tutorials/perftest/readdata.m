%%%based on the output of 
%%%grepshell.sh & multirun.sh

clear all;
close all;
clc ;

datadir = 'Job1/';
origdir = cd(datadir);

base='_688956.txt';

commADDTime = importdata(strcat('filtoutADDVAL',base));
commINSERTTime = importdata(strcat('filtoutINSERT',base));
NRanks  = importdata(strcat('nprocess',base));
MPIPackSizeADD = importdata(strcat('packsizeADDVAL',base));
MPIPackSizeINSERT = importdata(strcat('packsizeINSERT',base));
NodeNum  = importdata(strcat('cellnum',base));
CellNum = importdata(strcat('nodenum',base));
Overlap  = importdata(strcat('overlap',base));
VecDotT = importdata(strcat('vecdottime',base));
VecDotFlops = importdata(strcat('vecdotflops',base));
FEOrder = importdata(strcat('feorder',base));
NumComp  = importdata(strcat('compnum',base));
GVS= importdata(strcat('globvecsize',base));
SFPINSERT = importdata(strcat('sfpackINSERT',base));
SFUPINSERT = importdata(strcat('sfunpackINSERT',base));
SFPADDVAL = importdata(strcat('sfpackADDVAL',base));
SFUPADDVAL = importdata(strcat('sfunpackADDVAL',base));
SFMESSADDVAL  = importdata(strcat('sfmessADDVAL',base));
SFMESSINSERT  = importdata(strcat('sfmessINSERT',base));

