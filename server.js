
const express = require('express')
const {spawn} = require('child_process');
const cors = require('cors')
const bodyParser = require('body-parser');
const app = express()
const port = process.env.PORT;
app.use(express.urlencoded({extended: false}));
app.use(express.json());
app.use(cors())

app.use(express.static(__dirname + '/public'))



app.post('/', (req, res) => {
	  const { imageUrl } = req.body;
	 var dataToSend;
	 // spawn new child process to call the python script
	 const python = spawn('python', ['./python/myCnn.py',imageUrl]);
	 // collect data from script
	 python.stdout.on('data', function(data)  {
	  console.log('Pipe data from python script ...');
	  var prediction = data.toString().split(":")[1];
	  dataToSend = prediction;
 });


 python.stderr.on('data', (data) => {
  console.error(`stderr: ${data}`);
});

 // in close event we are sure that stream from child process is closed
 python.on('close', (code) => {
 console.log(`child process close all stdio with code ${code}`);
 // send data to browser
 res.send(dataToSend)
 });
 
})
app.listen(port, () => console.log(`Example app listening on port 
${port}!`))