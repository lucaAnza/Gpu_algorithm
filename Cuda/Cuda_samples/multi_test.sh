echo "Listing periferics :"

lspci | grep VGA

echo "Info on SCHEDA-VIDEO + Driver info : "

glxinfo | grep "OpenGL vendor"

echo "Status SCHEDA NVIDIA : "

nvidia-smi