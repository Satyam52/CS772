import React, { useEffect, useState } from "react";
import {
  MDBContainer,
  MDBRow,
  MDBCol,
  MDBCard,
  MDBCardHeader,
  MDBCardBody,
  MDBCardFooter,
  MDBIcon,
  MDBBtn,
  MDBTypography,
} from "mdb-react-ui-kit";
import './style.css'
import { MDBTable, MDBTableHead, MDBTableBody } from 'mdb-react-ui-kit';
import axios from "axios";

const Tables = ({scores}) => {
  const data = [
    {
      'activation': 'relu',
      'bert': '0.5',
      'bertTuned': '0.6'
    },
    {
      'activation': 'Prelu',
      'bert': '0.5',
      'bertTuned': '0.6'
    },
    {
      'activation': 'reluN',
      'bert': '0.5',
      'bertTuned': '0.6'
    },
    {
      'activation': 'Sigmoid',
      'bert': '0.5',
      'bertTuned': '0.6'
    }
  ]

  return (
    <MDBContainer  >
{Object.keys(scores).length>0 &&   
<>
 <MDBTypography tag='h1' >Scores on Bert base cased</MDBTypography>
  
  <MDBTable bordered borderColor="primary">
      <MDBTableHead>
        <tr>
          <th scope='col'><MDBTypography tag='b'>Activation</MDBTypography></th>
          <th scope='col'><MDBTypography tag='b'>Bert base cased</MDBTypography></th>
          {/* <th scope='col'><MDBTypography tag='b'>Bert fine-tuned</MDBTypography></th> */}
        </tr>
      </MDBTableHead>
      <MDBTableBody>
        {scores && Object.keys(scores).map((item,idx)=>{
          return (
            <tr key={idx}>
            <th scope="row">{item}</th>
            <td>{scores[item].toFixed(4)}</td>
            </tr>
          )
        })}

      </MDBTableBody>
    </MDBTable>
    </>}
    </MDBContainer>
  )
}


export default function App() {
  const [sentence, setSentence] = useState('')
  const [data, setData] = useState([])

  const onSend = async()=>{
    try {
        const res = await axios.post('http://localhost:8000/api/v1/activation',{'input':sentence})
        setData(res.data.scores)
        console.log(res.data.scores)
    } catch (error) {
        console.log(error)
    }
}


  return (
    <>
    <MDBContainer fluid className="py-5" style={{ backgroundColor: "#eee" }}>
      <MDBRow className="d-flex justify-content-center">
        <MDBCol md="10" lg="8" xl="6">
          <MDBCard id="chat2" style={{ borderRadius: "15px" }}>
            <MDBCardHeader className="d-flex justify-content-between align-items-center p-3">
                <h5 className="mb-0">SST-2 demo</h5>
              <MDBBtn onClick={()=>{
                setData([])
                setSentence(null)
              }} color="primary" size="sm" rippleColor="dark">
                Reset
              </MDBBtn>
            </MDBCardHeader>

            <MDBCardFooter className="text-muted d-flex justify-content-start align-items-center p-3 z-100">
              <input
                type="text"
                class="form-control form-control-lg"
                id="exampleFormControlInput1"
                placeholder="Give the input sentence here..."
                onChange={(e)=>setSentence(e.target.value)}
              ></input>
              <MDBBtn onClick={()=>onSend()} className="ms-3" href="#!">
                <MDBIcon fas icon="arrow-right" size='2x' />
              </MDBBtn>
            </MDBCardFooter>
          </MDBCard>
        </MDBCol>
      </MDBRow>
      </MDBContainer>

      <MDBContainer MDBContainer fluid className="py-5" >
        {/* <MDBTypography tag='h1'>Plots</MDBTypography> */}
        {/* <MDBContainer>
          <img src='https://mdbootstrap.com/img/new/slides/041.webp' className='img-fluid shadow-4 mb-10' alt='...' />
        </MDBContainer> */}
     
      <Tables scores={data}/>

      {/* <MDBTypography tag='h1' >Results on Distilbert</MDBTypography>
      <Tables /> */}
      </MDBContainer >
      </>
  );
}