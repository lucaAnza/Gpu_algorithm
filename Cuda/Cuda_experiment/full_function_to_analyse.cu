void Frame::ComputeStereoMatches()
{   

    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;  // Calcola una soglia

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;   // myImagePyramid è una lista di immagini a risoluzione sempre più basse  L[0] è l'immagine con qualità maggiore.


    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());   // Crea una matrice con nRows vettori 

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);   // Si prevedono almeno 200 punti chiave per ogni riga dell'immagine LEFT

    const int Nr = mvKeysRight.size();  // Numero punti chiave dell'immagine destra


    // Per ogni punto chiave dell'immagine a DESTRA si prende un raggio e si aggiungono tutti gli indici dei punti a "VRowIndices"
    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];

        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
        
    }


    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;


    // Chiama la funzione parallela di stereo matching   
    //TODO -> aggiungere vRowIndices ai parametri di ingresso!
    const cv::KeyPoint &kpLuke = mvKeys[0];
    const float x = kpLuke.pt.x;
    cout<<"Test cpu : "<<x<<endl;
    gpu_stereoMatches( mvKeys , minZ , minD , maxD );
    //gpu_stereoMatches( mvKeys , minZ , minD , maxD , ORBmatcher::TH_HIGH , mDescriptorsRight , mvInvScaleFactors , mpORBextractorLeft , mvScaleFactors );



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
            const cv::KeyPoint &kpR = mvKeysRight[iR];   // kpR assume il valore del punto corrispondente al candidato

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)  // kpR.octave rappresenta il livello piramidale (scala) del punto a DX, levelL di quello a SX
                continue;

            const float &uR = kpR.pt.x;  // coordinata x del punto candidato che stiamo analizzando

            if(uR>=minU && uR<=maxU)    // Controllo se la x del keypointCandidatoDX sta in un range
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);  
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);   // restituisce la distanza tra riga DX e SX (DA APPROFONDIRE)

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)    // vede se il punto migliore dei candidati supera una determinata soglia.
        {
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

            int bestDist = INT_MAX;
            int bestincR = 0;    // è il miglior spostamento della windows
            const int L = 5;
            vector<float> vDists;  // ha le distanze tra le finestre nelle immagini sx e dx per ogni possibile spostamento nell'intervallo da -L a +L
            vDists.resize(2*L+1);

            // calcolano i limiti della finestra scorrevole nella quale verrà effettuata la ricerca dei punti
            const float iniu = scaleduR0+L-w;       
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)   // per evitare di uscire dai range
                continue;

            // Si cerca il migliore incremento e la migliore distanza
            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);   // Esegue la norma1 tra la finestra_sx e la finestra_dx
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
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