#include <cstdlib>
#include <string>
#include "TDirectory.h"
#include "TFile.h"
#include "TKey.h"
#include <vector>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <filesystem>

/*
File contents:
	TpcSampaData.root
		TpcSampaData
			0 <- event
				hits <- std::vector<double>
				vertex <- std::vector<double>
			1
				hits
				vertex
			...
			1000
				hits
				vertex

	contents of hits std::vector
		[X Y Z Sector Row TimeBin nTracks TrackId MotherId Pt]
		X, Y, Z - hit global coordinates (cm)
		Sector - sector number
		Row - row number
		TimeBin - hit time from the start of the event (all Tpc -> 310 timebins, 1 timebin ~ 100 ns) 
		nTracks - number of track which hit belongs to
		after that, nTracks group of 4 numbers:
			TrackId - unique track id
			MotherId - track id from which track was born (-1 for primary tracks)
			Pt - pt momentum (GeV) per q
			
	contents of vertex std::vector
		X Y Z
		
*/

void SaveHits(TDirectory* pDir, std::string name) 
{
	std::filesystem::path dirPath = std::filesystem::path(name).parent_path();

	if (!std::filesystem::exists(dirPath)) 
    {
        if (!std::filesystem::create_directories(dirPath)) 
        {
            throw std::runtime_error("SaveHits: Fail to create the folder " + dirPath.string());
        }
    }

	TIter iKey(pDir->GetListOfKeys());
    while(TKey* pKey = (TKey*)iKey()) 
	{
		if(std::strstr(pKey->GetName(), "hits")) 
		{
			std::vector<std::vector<double>>* pVec;
			pDir->GetObject(pKey->GetName(), pVec);

			std::ofstream file_hits(name + "_hits.csv");
			std::ofstream file_truth(name + "_truth.csv");
			std::ofstream file_tracks(name + "_tracks.csv");

			file_hits << "hit_id,x,y,z,sector_id,row_id\n";
			file_truth << "hit_id,track_id\n";
			file_tracks << "track_id,mother_id,pt\n";

			std::unordered_set<size_t> processed_ids;

			for(int i=0; i<(*pVec).size(); i++)
			{ 			

				file_hits << i+1 << ",";
				
				for(int j=0; j<5; j++) 
				{
				    file_hits << (*pVec)[i][j] << ((j+1 == 5) ? "\n" : ",");
        	    }

				size_t k = 7;

				while(k < (*pVec)[i].size())
				{
					file_truth << i+1 << "," << (*pVec)[i][k] << "\n";

					if(processed_ids.find((*pVec)[i][k]) == processed_ids.end())
					{
						file_tracks << (*pVec)[i][k] << "," << (*pVec)[i][k+1] << "," << fabs((*pVec)[i][k+2]) << "\n";
						
						processed_ids.insert((*pVec)[i][k]);
					} 

					k += 4;
				}
			}

			file_hits.close();
			file_truth.close();
			file_tracks.close();

			processed_ids.clear();
		}
	}
}

void PrintHits(TDirectory* pDir) 
{
	TIter iKey(pDir->GetListOfKeys());
    while(TKey* pKey = (TKey*)iKey()) 
	{
		if(std::strstr(pKey->GetName(), "hits")) 
		{
			std::vector<std::vector<double>>* pVec;
			pDir->GetObject(pKey->GetName(), pVec);

			for(int i=0; i<20; i++)
			{ 			
        	    for (int j=0; j<(*pVec)[i].size(); j++) 
				{
				    std::cout << (*pVec)[i][j] << " ";
        	    }

        	    std::cout << std::endl;
			}

			std::cout << std::endl;
		}
	}
}

void PrintVertex(TDirectory* pDir) 
{
	TIter iKey(pDir->GetListOfKeys());

    while(TKey* pKey = (TKey*)iKey()) 
	{
		 if(std::strstr(pKey->GetName(), "vertex" )) 
		 {
			std::vector<double>* pVec;
			pDir->GetObject( pKey->GetName(), pVec );

			for( double x : *pVec)
			{ 				
				std::cout << x << " ";
			}
			std::cout << std::endl;
		 }
	}
}

void ReadHits(TDirectory* pDir, uint nEvent) 
{
	std::string strEvent = std::to_string(nEvent);
	TIter iKey(pDir->GetListOfKeys());
	
    while(TKey* pKey = (TKey*)iKey()) 
	{
        if(pKey->IsFolder() && !strcmp(pKey->GetName(), strEvent.c_str()))
		{			
			SaveHits(pDir->GetDirectory(pKey->GetName()), "./MPD_events/event_" + std::to_string(nEvent));
			//PrintHits(pDir->GetDirectory(pKey->GetName()));
			//PrintVertex(pDir->GetDirectory(pKey->GetName()));
		}
    }
}

void readSAMPA() 
{
    TFile* pFile = TFile::Open("TpcData_1000.root");

    if(pFile && pFile->IsOpen()) 
	{
        TIter iKey(pFile->GetListOfKeys());
        TKey* pKey;
		
        while((pKey = (TKey*)iKey()))
		{			
            if(pKey->IsFolder() && !strcmp(pKey->GetName(), "TpcEvents"))
			{
				for(size_t ev_num = 0; ev_num < 1000; ev_num++)
				{
				    ReadHits(pFile->GetDirectory(pKey->GetName()), ev_num);
				}

                break;
            } 
        }
    }
}