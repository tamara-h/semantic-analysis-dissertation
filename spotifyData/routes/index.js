const express = require('express');
const router = express.Router();
const topArtists = require('../models/dataModel');


const happy = {
    name: "Happiness",
    v: (4.53/5),
    a: (3.78/5),
    d: (3.65/5)
};

const anger = {
    name: "Anger",
    v: (1.23/5),
    a: (3.98/5),
    d: (3.13/5)
};

const disgust = {
    name: "Disgust",
    v: (1/5),
    a: (3.38/5),
    d: (2.78/5)
};

const fear = {
    name: "Fear",
    v: (0.9/5),
    a: (4/5),
    d: (1.43/5)
};
const sadness = {
    name: "Sadness",
    v: (0.93/5),
    a: (1.83/5),
    d: (1.68/5)
};
const surprise =  {
    name: "Surprise",
    v: (3.5/5),
    a: (4.18/5),
    d: (2.18/5)
}

let emotions = [happy, anger, disgust, fear, sadness, surprise];

/**
 * Get  /
 *
 */
router.get('/', function(req, res, next) {
  res.send({semanticSongs: {version: 0.01, dateMade: '27-02-2019'}});
});


/**
 *
 */
router.get('/valenceSong', function (req,res,next){
    console.log("GET Request received at /valenceSong");
    if(req.query.valence){
        let requestValence = req.query.valence;
        let requestArousal = req.query.arousal;
        let requestDominance = req.query.dominance;

        /**
         *
         * @param track
         * @returns {Promise<any>}
         */
        function getData(track){
            console.log("getting data");
            console.log(track);
            return new Promise((res,rej) => {
                // console.log(tracks[i].trackTitle);
                topArtists.getTrackInfo(track.sourceTrackID)
                    .then(trackInfo => {
                        // console.log(track);

                        let trackResponse = {
                            track: track.trackTitle,
                            // artist: track.artist.name,
                            trackUri: track.sourceTrackID,
                            valence: trackInfo.valence,
                            energy: trackInfo.energy,
                            danceability: trackInfo.danceability
                        };
                        res(trackResponse);

                    })
                    .catch(err => {
                        console.log(err);
                        rej(err);
                    })
            })
        }

        /**
         *
         * @param trackData
         * @param goal
         * @param dist
         * @returns {Promise<any>}
         */
        function getSimilarTrack(trackData, v,a,d){
            return new Promise((res, rej) => {

                distance = Math.pow((Math.pow((v-trackData.valence),2) + Math.pow((a-trackData.energy),2)) + Math.pow((d-trackData.danceability),2),0.5);

                res(distance);
            });
        }

        function calcEmotionDist(index, v,a,d){
            return new Promise((res,rej) => {
                dist = Math.pow((Math.pow((v-emotions[index].v),2) + Math.pow((a-emotions[index].a),2)) + Math.pow((d-emotions[index].d),2),0.5);
                console.log(index);
                console.log(dist);
                res(dist)
            })
        }

        function getSimilarEmotion(v,a,d){
            return new Promise((resolve,reject) => {
                shortestDist = 100;
                emotion = "ERROR";
                array = [];
                for (let i=0;i<emotions.length;i++){
                    array.push(calcEmotionDist(i,v,a,d));
                }
                Promise.all(array)
                    .then(res =>{
                        index = res.indexOf(Math.min(...res));
                        resolve(emotions[index].name);
                    })
                    .catch(err => {
                        reject(err);
                        console.log(err)
                    })

            })
        }

        /**
         *
         * @param tracks
         * @returns {Promise<any>}
         */
        function getTrackData(tracks){
            // valence = requestValence;
            return new Promise((resolve, reject) => {
                trackData = [];
                for (let i=0;i<tracks.length; i++){
                    trackData.push(getData(tracks[i]));
                }
                Promise.all(trackData)
                    .then(result => {
                        array = [];
                        distance = 2;
                        for (let j=0;j<result.length;j++){
                            array.push(getSimilarTrack(result[j], requestValence, requestArousal, requestDominance))
                        }
                        Promise.all(array)
                            .then(res =>{
                                index = res.indexOf(Math.min(...res));
                                resolve(result[index]);
                            })
                            .catch(err => {
                                reject(err);
                                console.log(err)
                            })



                    })
                    .catch(err => {
                        reject(err);
                    })
            });
        }

        topArtists.getTopTracks()
            .then(tracks => {
                console.log("Retrieved tracks successfully");
                let tracksAndIDs = [];
                for (let i=0; i<tracks.items.length; i++){
                    //I am aware that this is messy as hell. I'll fix this later
                    //TODO
                    actualId = tracks.items[i].uri.split('k:')[1];

                    tracksAndIDs.push({
                        "trackTitle": tracks.items[i].name,
                        "sourceTrackID": actualId,
                    });
                }
                getTrackData(tracksAndIDs)
                    .then(response => {
                        getSimilarEmotion(requestValence, requestArousal, requestDominance)
                            .then(mood => {
                                response.emotion = mood;
                                res.send(response)
                            })

                    })
                    .catch(err => {
                        res.send({Err: 'It would appear that this failed.'});
                    })

            })
            .catch( err => {
                console.error("Failed to get the artists");
                console.error(err);

                // Send on error to user
                res.status(500);
                res.send({Err: 'It would appear that this failed.'});
            });

    } else{
        res.status(500);
        res.send({Err: "Please suppy a valence"})
    }

});



router.get('/trackInfo', function (req,res){
    if(req.query.trackId) {
        let trackID = req.query.trackId;
        topArtists.getTrackInfo(trackID)
            .then(trackInfo => {
                console.log("got track info");
                res.send(trackInfo)
            })
            .catch(err =>{
                console.error('Failed to get track info');
                console.error(err);
                res.status(500);
                res.send({Err: "It would appear that getting the track info failed"})
            })
    }
    else{
        res.status(500);
        res.send({Err: "Please suppy a track Id"})
    }
});

router.get("/login", function(req,res){
    topArtists.auth()
        .then(authURL =>{
            res.send({authUrl: authURL})
        })
        .catch(err => {
            res.send("Something went wrong")
        })
    // let authURL = "https://accounts.spotify.com/authorize?client_id=2ef21279418442ca8807da295adbe1da&response_type=code&redirect_uri=http://localhost:8080/&scope=user-top-read&state=some-state-of-my-choice";
    // res.redirect(authURL);
});

router.get("/callback", function(req,res){
    console.log('at callback');
    authCode = req.query.code;
    // authCode.replace(/['"]+/g, '');
    console.log(authCode);
    topArtists.authorisationGrant(authCode)
        .then(data =>{
            res.redirect("http://localhost:8000/#!/view2");
        })
        .catch(err => {
            res.send("Something went wrong");
        })
});




router.get('/topTracks', function(req,res,next){
    console.log("GET Request Received at top");

    topArtists.getTopTracks()
        .then(tracks => {
            console.log("Retrieved getTopTracks successfully");
            let tracksAndIDs = [];
            // console.log(artists.items);
            for (let i=0; i<tracks.items.length; i++){
                artistsNames = "";
                // console.log(tracks.items[i].artists);
                for (let j=0; j<tracks.items[i].artists.length; j++){
                    artistsNames =  artistsNames + tracks.items[i].artists[j].name + " ";
                }
                tracksAndIDs.push({
                    "tracks": tracks.items[i].name,
                    "artists": artistsNames,
                    "uri": tracks.items[i].uri,
                    "popularity": tracks.items[i].popularity
                });
            }

            res.send({"items": tracksAndIDs});

        })
        .catch( err => {
            console.error("Failed to get the artists getTopArtists");
            console.error(err);

            // Send on error to user
            res.status(500);
            // res.send({Err: err});
            res.send({Err: 'It would appear that this failed. getTopArtists'});
        });
});


router.get("/createPlaylist", function(req,res){
    console.log('at create playlist');
    // authCode.replace(/['"]+/g, '');
    topArtists.createNewPlaylist()
        .then(data =>{
            console.log('success')
        })
        .catch(err => {
            console.error(err)
            res.send("Something went wrong");
        })
});




module.exports = router;