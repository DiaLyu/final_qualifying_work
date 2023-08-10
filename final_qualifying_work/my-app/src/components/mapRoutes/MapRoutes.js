import {useState, useEffect, useRef} from 'react';
import { MapContainer, TileLayer, Marker, Popup, ZoomControl, Polyline} from 'react-leaflet';
import L from "leaflet";

import RouteService from '../../service/RouteService';
import useLocationBrowser from '../../hooks/useLocationBrowser';
import LeafletRuler from './LeafletRuler';

import your_place from '../../resources/icons/your_place.png';

const MapRoutes = (props) => {
    const [points, setPoints] = useState([]);
    const [position, setPosition] = useState({lat: 55.7522, lng: 37.6156});                    // координаты Москвы
    const mapRef = useRef()

    const routeService = new RouteService();
    const location = useLocationBrowser();

    useEffect(() => {
        fetchBooks();
    }, [props.selectedInput])

    const fetchBooks = () => {
        const {selectedInput} = props;
        if (selectedInput.length === 0) {
            setPoints([])
            return;
        }
        for(let i = 0; i < selectedInput.length; i++){
            routeService.getRouteCharacter(selectedInput[i][0])
                .then(onBookListLoaded)
                .catch(onError)
        }
    }

    const onBookListLoaded = (newPoints) =>{
        let arr = []
        if (points.length === 0){
            arr.push(newPoints)
        }
        else{
            arr = points;
            console.log("arr1 ", arr)
            let flag = true;
            for(let i = 0; i < arr.length; i++){
                if(arr[i]["id"] === newPoints["id"]){
                    flag = false
                }
            }
            if (flag){
                arr.push(newPoints)
            }
        }

        setPoints(arr);
    }

    const onError = () => {
        console.log("error in getting routes");
    }

    const showMyLocation = () => {
        if (location.loaded && !location.error){
            mapRef.current.flyTo(
                [location.coords.lat, location.coords.lng],
                6,
                {animate: true}
            );
        } else {
            alert(location.error.message);
        }
    }

    const pointMarkers = (data) => {
        return data.map(item => {
            return (
                <>
                    <Marker key={item["order_in"]} position={[item["latitude"], item["longitude"]]}>
                        <Popup>
                            {item["name_place"]} <br /> {item["order_in"]}
                        </Popup>
                    </Marker>
                </>
            )
        })
    }

    const routes = () => {
        // console.log("MapRoutes selectedInput ", props.selectedInput)
        // console.log("MapRoutes points ", points)
        if (props.selectedInput.length !== 0){
            return points.map((item, i) => {
                if(props.selectedInput[i][1]){
                    let data = item["route"]
                    let route = []
                    for(let i = 0; i < data.length; i++){
                        let coords = [data[i]["latitude"], data[i]["longitude"]];
                        route.push(coords)
                    }
                    let markers = pointMarkers(data)
                    let colorRoute = "#" + Math.random().toString(16).substr(-6);
                    return (
                        <>
                            <Polyline key={item['id']} pathOptions={{color: colorRoute}} positions={route} />
                            {markers}
                        </>
                    )
                }
            })
        }
    }

    return (
        <>
            <MapContainer className="map" center={position} zoom={6} zoomControl={false} ref={mapRef}>
                <ZoomControl position='topright'/>
                <TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {routes()}
                {
                    location.loaded && !location.error && (
                        <Marker
                            icon={markerIcon}
                            position={[location.coords.lat, location.coords.lng]}
                        ></Marker>
                    )
                }
                <LeafletRuler/>
            </MapContainer>

            <div className="main__manage-map">
                <button className="main__manage-map__btn" onClick={showMyLocation}>
                    <img src={your_place} alt="Местоположение браузера" />
                </button>
            </div>
        </>
    )
}

const markerIcon = new L.Icon({
    iconUrl: require("../../resources/icons/markers/your_locat.png"),
    iconSize: [40, 45],
    iconAnchor: [17, 41],
    popupAnchor: [0, -40]
})

let DefaultIcon = L.icon({
    iconUrl: "/marker-icon.png",
    iconSize: [25, 41],
    iconAnchor: [10, 41],
    popupAnchor: [2, -40],
    });
    
L.Marker.prototype.options.icon = DefaultIcon;


export default MapRoutes;