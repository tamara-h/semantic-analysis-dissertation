const createError = require('http-errors');
const express = require('express');
const path = require('path');
const cookieParser = require('cookie-parser');
// const logger = require('morgan');
const serverless = require('serverless-http');

const indexRouter = require('./routes/index');
const songRouter = require('./routes/songDataRouter');
const app = express();


// app.use(logger('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

// Set headers
app.use(function (req, res, next) {

    res.setHeader('X-Powered-By', 'Tamara Power');
    res.setHeader('Access-Control-Allow-Origin', 'localhost');
    //Thats probably wrong

    next();

});


app.use('/', indexRouter);
app.use('/artists', songRouter);

// catch 404 and forward to error handler
app.use(function(req, res, next) {
  next(createError(404));
});

// error handler
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.send({error: err});
});


// Set up the serverless app
// module.exports.handler = serverless(app);
module.exports = app;
