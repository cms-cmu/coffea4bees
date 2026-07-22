OUTPUT_DIR=output

mkdir $OUTPUT_DIR

for file_name in hist__TTTo2L2Nu-UL16_postVFP.coffea \
		 hist__TTTo2L2Nu-UL16_preVFP.coffea \
		 hist__TTTo2L2Nu-UL17.coffea \
		 hist__TTTo2L2Nu-UL18.coffea \
		 hist__TTToHadronic-UL16_postVFP.coffea \
		 hist__TTToHadronic-UL16_preVFP.coffea \
		 hist__TTToHadronic-UL17.coffea	\
		 hist__TTToHadronic-UL18.coffea \
		 hist__TTToSemiLeptonic-UL16_postVFP.coffea \
		 hist__TTToSemiLeptonic-UL16_preVFP.coffea \
		 hist__TTToSemiLeptonic-UL17.coffea \
		 hist__TTToSemiLeptonic-UL18.coffea; do
   wget $REANA_OUTPUT_PATH/$file_name
   mv $file_name $OUTPUT_DIR/
done		 

#mv hist__TTTo2L2Nu-UL16_postVFP.coffea $OUTPUT_DIR/
#hist__TTTo2L2Nu-UL16_preVFP.coffea	    
