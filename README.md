# LeReNet

## Something-something training records

### Experiment 1

Models: I3D 32 frames input non-local and mask non-local. Use I3D non-local affine 32 frames inputs as pre-training model.


| <sub>Model</sub> | <sub>Test Acc</sub> | <sub>Commens</sub> | 
| ------------- | ------------- | ------------- |
| <sub>I3D in paper</sub> | <sub>41.6</sub> | <sub></sub> | 
| <sub>I3D non-local in paper</sub> | <sub>44.4</sub> | <sub></sub> | 
| <sub>I3D 32 input</sub> | <sub>38.7</sub> | <sub>2.9 drop than I3D</sub> | 
| <sub>I3D non-loal 32 input</sub> | <sub>44.05</sub> | <sub>0.35 drop than I3D nonlocal</sub> | 
| <sub>I3D mask nlnet 32 input</sub> | <sub>45.3</sub> | <sub>1.1 increase than I3D non-local</sub> | 
