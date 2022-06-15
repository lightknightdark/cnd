let model;

// webCam
const video = document.querySelector('video');

// webCam display
const canvas = document.getElementById('output');
const ctx = canvas.getContext('2d');

// debugMessage
const debugMessage = document.getElementById("debugMessage")
console.log("Width:", window.innerWidth)
console.log("Height:", window.innerHeight)

// stats library
const stats = new Stats();

const imgSize = 160
const modelUrlPath = 'https://cdn.jsdelivr.net/lightknightdark/project/models/model.json'
const [divNum , subNum] = [1,0] 





const labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z'] 



                async function getMedia() {
                    let stream = null;
                
                    let constraints = window.constraints = {
                        audio: false,
                        video: {
                            facingMode: "environment" 
                        }
                    };
                  
                    try {
                      stream = await navigator.mediaDevices.getUserMedia(constraints);
                      console.log(stream)
                
                      window.stream = stream
                      video.srcObject = stream
                
                    } catch(err) {
                        console.log(err);
                    }
                }
                
                
                // creata load model and active cameras
                async function loadModel(){
                
                    model = await tf.loadGraphModel(modelUrlPath);
                
                    // Set up canvas w and h
                    canvas.width = video.videoWidth;
                    canvas.height = video.videoHeight;
                
                    predictModel();
                
                    const capBtn = document.getElementById("capBtn");
                
                    capBtn.addEventListener("click", async () => {
                
                        let imgURL = canvas.toDataURL("image/png");
                
                        let dlLink = document.createElement('a');
                        dlLink.download = "fileName";
                        dlLink.href = imgURL;
                        dlLink.dataset.downloadurl = ["image/png", dlLink.download, dlLink.href].join(':');
                
                        document.body.appendChild(dlLink);
                        dlLink.click();
                        document.body.removeChild(dlLink);
                        
                    })
                
                }
                
                video.addEventListener('loadeddata', async () => {
                    console.log('Yay!');
                    loadModel();
                });
                
                window.onload = async () => {
                    stats.showPanel( 0 ); // 0: fps, 1: ms, 2: mb, 3+: custom
                    document.body.appendChild( stats.dom );
                
                    getMedia();
                }
                
                var requestAnimationFrameCross = window.webkitRequestAnimationFrame ||
                        window.requestAnimationFrame || window.mozRequestAnimationFrame ||
                        window.oRequestAnimationFrame || window.msRequestAnimationFrame;
                
                async function predictModel(){
                    
                    stats.begin();
                
                    // Prevent memory leaks by using tidy 
                    let imgPre = await tf.tidy(() => {
                        return tf.browser.fromPixels(video)
                                .resizeNearestNeighbor([imgSize, imgSize])
                                .toFloat()
                                .div(tf.scalar(divNum)) 
                                .sub(tf.scalar(subNum))
                                .expandDims();
                    });
                    
                    const result = await model.predict(imgPre).data();
                
                    await tf.dispose(imgPre); // clear memory
                
                    let ind = result.indexOf(Math.max(...result));
                    //console.log("MyModel predicted:", labels[ind]); // top labels
                    //console.log("Possibility:", result[ind] * 100); // top leabels possible
                
                    ctx.drawImage(video, 0, 0);
                
                    // Draw the top color box
                    ctx.fillStyle = "#00FFFF";
                    ctx.fillRect(0, 0, 1000, 30);
                    
                    // Draw the text last to ensure it's on top. (draw label)
                    const font = "22px sans-serif";
                    ctx.font = font;
                    ctx.textBaseline = "top";
                    ctx.fillStyle = "#000000";
                    ctx.fillText(`${labels[ind]} : ${result[ind] * 100}%`, 20, 8); 
                
                    stats.end();
                    requestAnimationFrameCross(predictModel);        
                }
                
                