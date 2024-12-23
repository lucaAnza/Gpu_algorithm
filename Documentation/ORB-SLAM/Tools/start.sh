command="nvidia-smi"
match_str="NVIDIA-SMI"
res=$($command | grep -o $match_str)

stringa1="a"
stringa2="b"

# Confronta le stringhe
if [ "$res" = "$match_str" ]; then
  cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release

    make -j4

    cd ..

    Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI04-12.yaml ~/Desktop/dataset/sequences/07
else
  echo "Errore: possibili problemi con la Gpu!"
  echo "Log error : "
  $command
fi



