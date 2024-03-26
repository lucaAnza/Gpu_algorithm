#Questo script se avviato nella cartella principale /ORB_SLAM3
#Esegue il programma ed avvia l'esempio n.7


arg="$1"

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

if [ "$arg" -eq 1 ]; then
    make
else
    make -j16
fi

cd ..

Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml ~/Desktop/dataset/sequences/07
