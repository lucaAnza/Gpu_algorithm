int time_calls = 0;   //(luke_add)

void Frame::ComputeStereoMatches()
{      
    time_calls++;
    

    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;  // Calcola una soglia

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;   // myImagePyramid è una lista di immagini a risoluzione sempre più basse  L[0] è l'immagine con qualità maggiore.

    // Creazione array riferimento per GPU (luke_add)
    vector<size_t> size_refer;   // Array in cui ogni elemento c'è il numero di colonne per ogni riga di vRowIndices  //(luke_add)
    size_refer.resize(nRows);

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());   // Crea una matrice con nRows vettori 

    for(int i=0; i<nRows; i++){
        vRowIndices[i].reserve(200);   // Si prevedono almeno 200 punti chiave per ogni riga dell'immagine LEFT
        size_refer[i] = 0;  // Init of the array (luke_add)
    }

    const int Nr = mvKeysRight.size();  // Numero punti chiave dell'immagine destra


    // Per ogni punto chiave dell'immagine a DESTRA si prende un raggio e si aggiungono tutti gli indici dei punti a "VRowIndices"
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];

        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++){
            vRowIndices[yi].push_back(iR);
            size_refer[yi]++;   //(luke_add)
        }
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    //ATTEMPT
    //uchar* testArr = mpORBextractorLeft->getd_images();
    //printf("DFSF= %u\n" , testArr[0]);
    //printf("DFSF= %u\n" , testArr[1]);
    
    vector<int> best_dist_line_iL;  // (luke add)
    vector<size_t> best_dist_line_index_iL;  // (luke add)

    // For each left keypoint search a match in the right image  -> I candidati possibili sono nel vettore "vRowIndices -> vCandidates"
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);
    
    for(int iL=0; iL<N; iL++)          // Iterazione dei punti chiave SX
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];   // Init dei candidati (i candidati sono i punti destra che sono in un intorno di quelli a sx)

        if(vCandidates.empty())     // Caso in cui in cui nell'immagine a destra ci sono zero punti chiave candidati.
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL); // dL -> descriptor left assume il valore della riga iL

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)           // Iterazione dei punti chiave a DX (Candidati)
        {
            
            const size_t iR = vCandidates[iC];     // iR assume l'indice di ogni candidato


            //printf("{%d}[CPU]element of mvKeys[iL].pt.y(vL) : %f ,  iL[%d] iR[%lu] take index : %lu  \n" , time_calls , mvKeys[iL].pt.y  ,iL , iR , iC ); 
            const cv::KeyPoint &kpR = mvKeysRight[iR];   // kpR assume il valore del punto corrispondente al candidato

            //printf("{%d}[CPU] going to calculate dist of element iL[%d] iR[%lu] , num-elem = %d \n" , time_calls , iL , iR , vCandidates.size() );

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)  // kpR.octave rappresenta il livello piramidale (scala) del punto a DX, levelL di quello a SX
                continue;

            const float &uR = kpR.pt.x;  // coordinata x del punto candidato che stiamo analizzando

            if(uR>=minU && uR<=maxU)    // Controllo se la x del keypointCandidatoDX sta in un range
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);  
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);   // restituisce la distanza tra riga DX e SX (DA APPROFONDIRE)
                //printf("{%d}[CPU] distance of element iL[%d] iR[%lu] : %d , num-elem = %d \n" , time_calls , iL , iR , dist, vCandidates.size() );
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        //printf("{%d} [CPU] Distanza minimima della linea iL(%d) = %d\n" , time_calls , iL , bestDist);
        // Add this array for testing the accuracy on GPU
        best_dist_line_iL.push_back(bestDist);
        best_dist_line_index_iL.push_back(bestIdxR);

        // Subpixel match by correlation
        if(bestDist<thOrbDist)    // vede se il punto migliore dei candidati supera una determinata soglia.
        {   
            //printf("{%d}CPU iL = %d kpl.octave : %d , size of the piramid : h=%d , w=%d \n" , time_calls ,iL , kpL.octave , mpORBextractorLeft->mvImagePyramid[kpL.octave].size().height , mpORBextractorLeft->mvImagePyramid[kpL.octave].size().width);
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;        // Prende il valore della x del miglior candidato tra i KeyPoint_Right
            const float scaleFactor = mvInvScaleFactors[kpL.octave];   // Ottiene la scaleFactor da KeyPoint_Left
            const float scaleduL = round(kpL.pt.x*scaleFactor);     // x del KeyPoint_Left ridimensionata
            const float scaledvL = round(kpL.pt.y*scaleFactor);     // y del KeyPoint_Left ridimensionata
            const float scaleduR0 = round(uR0*scaleFactor);         // x del KeyPoint_Right ridimensionata

            // sliding window search
            const int w = 5;
            // Estrae una sottomatrice per il KeyPoint_Left
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1); 

            /*
            //(luke_add) IF Created for debug (iL == -1 to disable debug)
            if(iL == 421 || iL == 422 | iL == 899 | iL == 900 | iL == 3 | iL == 4){
                int rows = mpORBextractorLeft->mvImagePyramid[kpL.octave].size().height;
                int cols = mpORBextractorLeft->mvImagePyramid[kpL.octave].size().width;
                rows = 10; // FOR TESTING - original for should be from i -> rows
                cols = 10; // FOR TESTING - original for should be from j -> cols
                for (int i=0 ; i<rows ; i++){
                    for(int j=0 ; j<cols ; j++){
                        int index = (i*cols) + j;
                        printf("{%d}\tCPU - iL[%d] - mvImagePyramid[%d] array of size[%d][%d] = [%d][%d] : %u \n" ,time_calls , iL , kpL.octave , rows,cols,i,j, mpORBextractorLeft->mvImagePyramid[kpL.octave].at<uchar>(i,j));   
                    }
                }
            } //////// Finish IF debug
            */

            int bestDist = INT_MAX;
            int bestincR = 0;    // è il miglior spostamento della windows
            const int L = 5;
            vector<float> vDists;  // ha le distanze tra le finestre nelle immagini sx e dx per ogni possibile spostamento nell'intervallo da -L a +L
            vDists.resize(2*L+1);

            // calcolano i limiti della finestra scorrevole nella quale verrà effettuata la ricerca dei punti

            

            const float iniu = scaleduR0+L-w;       
            const float endu = scaleduR0+L+w+1;

            //printf("{%d}CPU iL[%d] PRE-FILTER iniu = %f , endu = %f FILTER = %u \n" , time_calls , iL , iniu , endu , mpORBextractorRight->mvImagePyramid[kpL.octave].cols) ;
            //printf("scaledvL = %f , scaleduR0 = %f \n" ,scaledvL , scaleduR0);

            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)   // per evitare di uscire dai range
                continue;

            // Si cerca il migliore incremento e la migliore distanza
            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                if(iL == 3){
                    printf("{%d} comparison(IL-IR) iL = 3  incR = %d: \n" , time_calls ,incR);
                    for (int i = 0; i < IL.rows; ++i) {
                        for (int j = 0; j < IL.cols; ++j) {
                            // Se l'immagine è a 1 canale (grayscale)
                            if (IL.channels() == 1) {
                                std::cout << (int)IL.at<uchar>(i, j) << "-";
                                std::cout << (int)IR.at<uchar>(i, j) << " ";
                            }
                            // Se l'immagine è a 3 canali (RGB)
                            else if (IL.channels() == 3) {
                                cv::Vec3b pixel = IL.at<cv::Vec3b>(i, j);
                                std::cout << "(" << (int)pixel[0] << ", " << (int)pixel[1] << ", " << (int)pixel[2] << ") ";
                            }
                        }
                        std::cout << std::endl;
                    }
                }


                float dist = cv::norm(IL,IR,cv::NORM_L1);   // Esegue la norma1 tra la finestra_sx e la finestra_dx
                
                if(iL == 3)
                    printf("CPU {%d} incr(%d) norma1 = %f\n\n" , time_calls , incR , dist);

                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(iL == 3){
                for(int i=0 ; i<(2*L+1) ; i++){
                    printf("CPU {%d} VDIST(%d)  =  %f  \n" , time_calls , i ,vDists[i] );  
                }
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    // Chiama la funzione parallela di stereo matching     (luke_add)
    gpu_stereoMatches( mpORBextractorLeft , mpORBextractorRight , time_calls , vRowIndices ,mvKeys , mvKeysRight , minZ , minD , maxD , ORBmatcher::TH_HIGH ,thOrbDist , mDescriptors , mDescriptorsRight , mvInvScaleFactors , mvScaleFactors , size_refer , best_dist_line_iL ,  best_dist_line_index_iL);

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}
