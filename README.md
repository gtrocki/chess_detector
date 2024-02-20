# chess_detector

Description:
------------
DL model for detecting the state of an arbitrary board of chess from an image
Fuller description in the proposal for the project: https://www.overleaf.com/7477313623gwxnfkvcgphb#e604bf (carefull with this link - everyone with it can edit the abstract!)

Possible resources:
-------------------
1. Repository of Masouris code: https://github.com/ThanosM97/end-to-end-chess-recognition
2. Dataset corresponding to the Masouris repository (chessReD): https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f/2
3. trained ChessCog model: https://github.com/georg-wolflein/chesscog

Interesting things to do/add to the model:
------------------------------------------
  * Try to make the detection transformer (DETR) work.
  * Simply fine-tune the model or change something in the architecture to get a higher success rate.
  * Implement the few-shot approach (as in Wölflein and Arandjelović, resource #3) to the Masouris model (resource #1)
     - This is probably more challenging than it appears, but possibly a type of transfer learning can be employed. Something like using the trained ResNeXt model of Masouris, but re-training only the layers that notice the chess pieces with the few-shot approach of showing it 2/3 images of the unseen board.
  * If actually adding to the model fails: try to investigate what each layer learns to detect (and maybe why?)
