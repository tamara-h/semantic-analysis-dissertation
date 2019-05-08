
angular.module('myApp.view1', ['ngRoute'])

.config(['$routeProvider', function($routeProvider) {
  $routeProvider.when('/view1', {
    templateUrl: 'view1/view1.html',
    controller: 'View1Ctrl'
  });
  $routeProvider.when('/callback', {
      templateUrl: 'view1/view1.html',
      controller: 'View1Ctrl'
  });
}])

.controller('View1Ctrl', function($window, $scope, $location, $routeParams, $http, $sce) {

    $scope.textboxhere = "Get creative!";
    $scope.songBaseURL = "https://open.spotify.com/embed/track/";
    $scope.songURI = "";
    $scope.songUrl = $sce.trustAsResourceUrl($scope.songBaseURL);
    $scope.loggedIn = false;


    $scope.login = function(){
        let loginURL = "http://localhost:8080/login";
        $http({
            method: 'GET',
            url: loginURL
        }).then(function successCallback(response) {
            if(response.status == 200){
                $window.location.href = response.data.authUrl;
            }

        }, function errorCallback(response) {
            console.error(response);
        });
    };




});