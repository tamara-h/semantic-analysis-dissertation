const express = require('express');
const router = express.Router();

const topArtists = require('../models/dataModel');


/**
 * GET /concerts
 */
router.get('/playlist', function (req, res, next) {

    // Get the Spofity API
    console.log("GET Request Received at artist/playlist");

    function getData(track){
        return new Promise((res,rej) => {
            // console.log(tracks[i].trackTitle);
            topArtists.getTrackInfo(track.sourceTrackID)
                .then(trackInfo => {
                    // console.log(trackInfo)
                    let trackResponse = {
                        trackTitle: track.trackTitle,
                        valence: trackInfo.valence
                    };
                    res(trackResponse);

                })
                .catch(err => {
                    console.log(err);
                    rej(err);
                })
        })
    }

    function getTrackData(tracks){
        // console.log(tracks.length);
        return new Promise((resolve, reject) => {
            trackData = [];
            for (let i=0;i<tracks.length; i++){
                trackData.push(getData(tracks[i]));
            }
            Promise.all(trackData)
                .then(res => {
                    resolve(res);
                })
                .catch(err => {
                    reject(err);
                })
        });
    }

    topArtists.getPlaylist()
        .then(tracks => {
            console.log("Retrieved artists successfully");
            getTrackData(tracks)
                .then(response => {
                    res.send(response)
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

});

router.get('/topArtists', function(req,res,next){
    console.log("GET Request Received at top");

    topArtists.getTopArtists()
        .then( artists => {
            console.log("Retrieved getTopArtists successfully");
            let artistsAndIds = [];
            // console.log(artists.items);
            for (let i=0; i<artists.items.length; i++){
                artistsAndIds.push({"artist": artists.items[i].name,
                                    "uri": artists.items[i].uri});
            }
            // console.log(artistsAndIds);

            res.send({"items": artistsAndIds});

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


module.exports = router;
