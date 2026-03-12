#include "captive_portal.h"

#include <ESP8266WiFi.h>
#include <DNSServer.h>
#include <ESP8266WebServer.h>

DNSServer dnsServer;
ESP8266WebServer server(80);

void startCaptivePortal(float spo2,float altitude,String risk){

  WiFi.softAP("TrekGuardian_Rescue");

  IPAddress myIP = WiFi.softAPIP();

  dnsServer.start(53,"*",myIP);

  server.on("/",[spo2,altitude,risk](){

    String page="<html><body>";
    page+="<h2>Trek Guardian Rescue</h2>";
    page+="SpO2:"+String(spo2)+"<br>";
    page+="Altitude:"+String(altitude)+"<br>";
    page+="Risk:"+risk;
    page+="</body></html>";

    server.send(200,"text/html",page);

  });

  server.begin();
}

void handlePortal(){
  dnsServer.processNextRequest();
  server.handleClient();
}
