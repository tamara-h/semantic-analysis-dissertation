
angular.module('myApp.view2', ['ngRoute'])

    .config(['$routeProvider', function($routeProvider) {
        $routeProvider.when('/view2', {
            templateUrl: 'view2/view2.html',
            controller: 'view2Ctrl'
        });
        $routeProvider.when('/callback', {
            templateUrl: 'view2/view2.html',
            controller: 'view2Ctrl'
        });
    }])

    .controller('view2Ctrl', function($window, $scope, $location, $routeParams, $http, $sce) {

        $scope.textboxhere = "";
        $scope.songURI = "";
        $scope.songBaseURL = "https://open.spotify.com/embed/track/";
        $scope.songUrl = $sce.trustAsResourceUrl($scope.songBaseURL);
        $scope.loggedIn = false;
        $scope.moodShown = false;
        $scope.v = "Valence: ";
        $scope.a = "Arousal: ";
        $scope.d = "Dominance: ";

        // var randomSentences = ["The winters ache in the icelandic twilight, we can't move away from the inevitable march." +
        // " But for now the air is glass and the fur of the stars encompass our tired bodies. I am so happy that you are here.", "Ginger nut biscuits after a roast dinner at a pub. Now that's what I can call fun!" +
        // " We'll head on over to the park afterwards, maybe Jackie and the boys will be there, I haven't seen her in ages.", "Let's got somewhere new. I mean I'm going, you can come with me if you'd like but I'm still going. I need a change and this will give me new insight, " +
        // "it'll be an adventure! You can come with me, but only if you want to. I'd like it if you came, but I'd honestly understand if you stayed.", "the fish in the water surrounding me weave between venetian blind cities. I think in binary colours, analzying the construction of the words on a screen.", "I'm so upset my presentation went badly"];



        var randomSentences = ["The winters ache in the icelandic twilight, we can't move away from the inevitable march." +
        " But for now the air is glass and the fur of the stars encompass our tired bodies. I am so happy that you are here.",
            "I'm sad. I feel very worried.", "I'm having the best day ever, I love my life and my wife and kids",
            "So far this week I have been writing the report for my final year project on the semantic analysis of text. I have enjoyed doing my project and am finding writing up all the work I have done very satisfactory. I am a little tired, but that is to be expected! "];
        $scope.randomText = function(){
            var rand = randomSentences[Math.floor(Math.random() * randomSentences.length)];
            $scope.textboxhere = rand;
        };
        $scope.randomText();


        $scope.login = function(){
            let loginURL = "http://localhost:8080/login";
            $http({
                method: 'GET',
                url: loginURL
            }).then(function successCallback(response) {
                if(response.status == 200){
                    $window.location.href = response.data.authUrl;
                }
                // console.log(response);
                // console.log(response.data);
                // alert(response.data);
                //
            }, function errorCallback(response) {
                console.error(response);
                // called asynchronously if an error occurs
                // or server returns response with an error status.
            });
        };

        $scope.showMood = function(){
            $scope.moodShown = !$scope.moodShown;
        }


        $scope.sendText = function () {
            // $scope.$apply();

            let emotiveTextURL = "http://localhost:8090/getVAD";

            if($scope.textboxhere != ""){
                emotiveTextURL = "http://localhost:8090/getVAD?words='" + $scope.textboxhere + "'";
            }



            $http({
                method: 'GET',
                url: emotiveTextURL
            }).then(function successCallback(response) {
                // console.log(response);
                // console.log(response.data);
                let valence = response.data.v;
                let arousal = response.data.a;
                let dominance = response.data.d;

                $scope.v = "Valence: " + valence;
                $scope.a = "Arousal: " + arousal;
                $scope.d = "Dominance: "+ dominance;
                $scope.mood = "Loading...";

                console.log("Valence: " + valence);
                console.log("Arousal: " + arousal);
                console.log("Dominance: "+ dominance);


                $http({
                    method: 'GET',
                    url: 'http://localhost:8080/valenceSong?valence=' + valence + "&dominance=" + dominance + "&arousal=" + arousal
                }).then(function successCallback(response) {
                    console.log(response.data);

                    $scope.mood = response.data.emotion;

                    $scope.output = response.data.track + " " + response.data.artist;
                    $scope.songURI = response.data.trackUri;

                    $scope.songUrl = $sce.trustAsResourceUrl($scope.songBaseURL + $scope.songURI);
                    $scope.loggedIn = true;
                    // this callback will be called asynchronously
                    // when the response is available
                }, function errorCallback(response) {
                    console.error(response);

                    // called asynchronously if an error occurs
                    // or server returns response with an error status.
                });




                // this callback will be called asynchronously
                // when the response is available
            }, function errorCallback(response) {
                console.error(response);
                // called asynchronously if an error occurs
                // or server returns response with an error status.
            });
        };


    });