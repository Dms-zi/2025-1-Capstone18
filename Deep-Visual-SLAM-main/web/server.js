// var express = require('express');
// var app = express();
// var cors = require('cors');

// let fs = require('fs');
// let options = {
//     key: fs.readFileSync('./tsp-xr.com_20241010671FC.key.pem'),
//     cert: fs.readFileSync('./tsp-xr.com_20241010671FC.crt.pem'),
//     requestCert: false,
//     rejectUnauthorized: false
// };
// app.set('view engine', 'ejs');
// app.engine('html', require('ejs').renderFile);

// app.use(cors());
// console.log(__dirname);
// app.use('/assets', express.static(__dirname + '/assets'));
// app.use('/styles', express.static(__dirname + '/styles'));
// app.use('/modules', express.static(__dirname + '/modules'));
// app.use('/build', express.static(__dirname + '/node_modules/three/build'));
// app.use('/gltf', express.static(__dirname + '/node_modules/three/'));
// var server_port = 5555;
// var server = require('https').createServer(options, app);

// app.get('/', (req, res) => {
  
//     res.render(__dirname + "/cam.html");    // index.ejs을 사용자에게 전달
// })

// server.listen(server_port, function() {
//   console.log( 'Express server listening on port ' + server.address().port );
// });
const { exec } = require('child_process');
const express = require('express');
const app = express();
const cors = require('cors');
const fs = require('fs');   
const https = require('https');
const path = require('path');
const multer = require('multer');

// HTTPS 옵션 (인증서 경로는 실제 파일명에 맞게 수정)
const options = {
    key: fs.readFileSync('./localhost+2-key.pem'),
    cert: fs.readFileSync('./localhost+2.pem'),
    rejectUnauthorized: false, // self-signed 인증서 사용 시 false로 설정
    requestCert: false 
};

app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);

app.use(cors());

console.log(__dirname);
app.use('/assets', express.static(path.join(__dirname, 'assets')));
app.use('/styles', express.static(path.join(__dirname, 'styles')));
app.use('/modules', express.static(path.join(__dirname, 'modules')));
app.use('/build', express.static(path.join(__dirname, 'node_modules/three/build')));
app.use('/gltf', express.static(path.join(__dirname, 'node_modules/three/')));

const server_port = process.env.PORT || 5555;

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, "last.html"));
});

app.get('/last', (req, res) => {
    res.sendFile(path.join(__dirname, "last.html"));
});

// 업로드 저장 경로 동적 생성
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        // 클라이언트에서 받은 seq 값 사용, 없으면 'sequence_01' 기본값
        const seq = req.body.seq || 'sequence_01';
        const uploadPath = path.join(__dirname, '../vo/last', seq);
        fs.mkdirSync(uploadPath, { recursive: true });
        cb(null, uploadPath);
    },
    filename: function (req, file, cb) {
        cb(null, file.originalname); // 00000.jpg, 00001.jpg, ...
    }
});
const upload = multer({ storage: storage });

// 업로드 라우트
app.post('/upload_sequence', upload.array('frames', 10000), (req, res) => {
    // 업로드 성공 응답을 먼저 보냄
    res.json({ success: true, files: req.files.length });

    // predict.py를 백그라운드에서 실행 (응답과 무관하게)
    exec('xvfb-run -a python3 ../vo/predict.py', (error, stdout, stderr) => {
        if (error) {
            console.error(`predict.py 실행 오류: ${error}`);
            console.error(stderr);
        } else {
            console.log(`predict.py 결과: ${stdout}`);
        }
    });
});

app.use(express.static(path.join(__dirname))); // 정적 파일 제공
app.use('/results', express.static(path.join(__dirname, '../vo/results')));


// HTTPS 서버로 실행
https.createServer(options, app).listen(server_port, '0.0.0.0', function() {
    console.log('Express HTTPS server listening on port ' + server_port);
    console.log('  Local: https://localhost:' + server_port);
    console.log('  Network: https://YOUR_SERVER_IP:' + server_port);
});
